"""Corner Shots Calculator CLI entry point."""

import argparse
import logging
import sys
from typing import Optional

from src.api_client import StatsAPIClient, StatsAPIError
from src.feature_engineering import extract_advanced_features
from src.models.bayesian_model import BayesianHierarchicalEngine
from src.models.glm_poisson import GLMCountModel
from src.telegram_bot import send_telegram_alert
from src.value_finder import ValueEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def str2bool(value: str) -> bool:
    """Convert a string argument to a boolean.

    Args:
        value: String representation of a boolean.

    Returns:
        Parsed boolean.

    Raises:
        argparse.ArgumentTypeError: If the value is not a recognised boolean string.
    """
    if isinstance(value, bool):
        return value
    lower = value.lower()
    if lower in {"yes", "true", "t", "y", "1"}:
        return True
    if lower in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected. Got '{value}'."
    )


def format_report(
    home: str,
    away: str,
    live: bool,
    minute: int,
    home_feat: dict[str, float],
    away_feat: dict[str, float],
    bayesian_corners: float,
    total_variance: float,
    expected_shots_tot: float,
    expected_shots_target: float,
    prob_over: float,
    target_line: float,
    time_decay: float,
    bet_result: Optional[dict[str, float | bool]] = None,
    odds: float = 0.0,
) -> str:
    """Build the Markdown report string.

    Args:
        home: Home team name.
        away: Away team name.
        live: Whether the match is in-play.
        minute: Current minute (if live).
        home_feat: Home team weighted features.
        away_feat: Away team weighted features.
        bayesian_corners: Posterior corner expectation.
        total_variance: Combined variance estimate.
        expected_shots_tot: Expected total shots.
        expected_shots_target: Expected shots on target.
        prob_over: Probability of exceeding the line.
        target_line: Over/under threshold.
        time_decay: Temporal decay factor applied.
        bet_result: Optional value-engineering result.
        odds: Market odds (if provided).

    Returns:
        Formatted Markdown report.
    """
    live_status = f"🔴 EN VIVO (Minuto {minute})" if live else "📅 PRE-PARTIDO"

    lines = [
        f"🤖 *QUANT ENGINE V2 REPORT* | {live_status}",
        f"⚽ *{home.upper()} vs {away.upper()}*",
        "",
        "📊 *MÉTRICAS PREDICTIVAS (Ponderadas):*",
        f"• xT Local: `{home_feat.get('xT', 0.0):.2f}` | Visitante: `{away_feat.get('xT', 0.0):.2f}`",
        f"• RoT (Precisión Tiro): `{home_feat.get('RoT', 0.0) * 100:.1f}%` / `{away_feat.get('RoT', 0.0) * 100:.1f}%`",
        f"• PPDA (Presión): `{home_feat.get('PPDA', 12.0):.1f}` / `{away_feat.get('PPDA', 12.0):.1f}`",
        f"• CII (Juego Bandas): `{home_feat.get('CII', 0.0):.1f}` / `{away_feat.get('CII', 0.0):.1f}`",
        "",
        "🎯 *PROYECCIONES RESTANTES EN EL ENCUENTRO:*",
        f"• Córneres Esperados: `{bayesian_corners:.2f}` (Var: {total_variance:.2f})",
        f"• Remates Totales: `{expected_shots_tot * time_decay:.2f}`",
        f"• Remates al Arco: `{expected_shots_target * time_decay:.2f}`",
        f"📈 *Probabilidad Over {target_line}:* `{prob_over * 100:.2f}%`",
    ]

    if bet_result is not None and odds > 1.0:
        lines.extend([
            "",
            f"💰 *EVALUACIÓN DE CUOTA ({odds})*",
            f"• EV+: `{bet_result['ev_percentage']}%`",
            f"• ¿Hay Valor?: `{'SÍ' if bet_result['has_value'] else 'NO'}`",
            f"• Stake Sugerido (Kelly): `{bet_result['recommended_stake']} / 3`",
        ])

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Execute the full prediction pipeline.

    Args:
        argv: Optional CLI argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description="Quantitative corner-kick and shot prediction engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--match-id", default="", help="Official match ID (optional)")
    parser.add_argument(
        "--odds-corners",
        type=float,
        default=0.0,
        help="Market over-line odds (e.g. 1.85)",
    )
    parser.add_argument(
        "--live",
        type=str2bool,
        default=False,
        help="Is the match currently in-play?",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=0,
        help="Current minute (required if --live is True)",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Resolve team identities
    # ------------------------------------------------------------------
    try:
        client = StatsAPIClient()
    except StatsAPIError as exc:
        logger.error("API client initialisation failed: %s", exc)
        return 1

    try:
        home_id = client.search_team_id(args.home)
        away_id = client.search_team_id(args.away)
    except StatsAPIError as exc:
        logger.error("Team resolution failed: %s", exc)
        return 1

    logger.info("Resolved IDs — %s: %d | %s: %d", args.home, home_id, args.away, away_id)

    # ------------------------------------------------------------------
    # 2. Fetch historical data
    # ------------------------------------------------------------------
    try:
        home_matches = client.get_last_n_matches_stats(home_id)
        away_matches = client.get_last_n_matches_stats(away_id)
    except StatsAPIError as exc:
        logger.error("Failed to retrieve match history: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    home_features = extract_advanced_features(home_matches, home_id)
    away_features = extract_advanced_features(away_matches, away_id)

    if not home_features or not away_features:
        logger.error("Feature extraction returned empty metrics for one or both teams.")
        return 1

    # ------------------------------------------------------------------
    # 4. Statistical modelling
    # ------------------------------------------------------------------
    try:
        bayesian_corners = BayesianHierarchicalEngine.posterior_corners_estimate(
            home_features, away_features
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Bayesian estimation failed: %s", exc)
        return 1

    expected_shots_total = (
        home_features.get("weighted_shots_total", 0.0)
        + away_features.get("weighted_shots_total", 0.0)
    )
    expected_shots_target = (
        home_features.get("weighted_shots_target", 0.0)
        + away_features.get("weighted_shots_target", 0.0)
    )
    total_variance = (
        home_features.get("var_corners", 1.0)
        + away_features.get("var_corners", 1.0)
    )

    target_line = 9.5
    time_decay = 1.0

    # ------------------------------------------------------------------
    # 5. In-play temporal adjustments
    # ------------------------------------------------------------------
    if args.live and args.minute > 0:
        time_decay = max(0.0, (90 - args.minute) / 90.0)
        bayesian_corners *= time_decay
        total_variance *= time_decay

    # ------------------------------------------------------------------
    # 6. Count-model probability
    # ------------------------------------------------------------------
    try:
        prob_over = GLMCountModel.predict_negative_binomial(
            lambda_exp=bayesian_corners,
            variance=total_variance,
            line=target_line,
        )
    except Exception as exc:
        logger.error("GLM prediction failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # 7. Value analysis
    # ------------------------------------------------------------------
    bet_result: Optional[dict[str, float | bool]] = None
    if args.odds_corners > 1.0:
        try:
            bet_result = ValueEngine.evaluate_bet(prob_over, args.odds_corners)
        except ValueError as exc:
            logger.warning("Value evaluation skipped: %s", exc)

    # ------------------------------------------------------------------
    # 8. Reporting
    # ------------------------------------------------------------------
    report = format_report(
        home=args.home,
        away=args.away,
        live=args.live,
        minute=args.minute,
        home_feat=home_features,
        away_feat=away_features,
        bayesian_corners=bayesian_corners,
        total_variance=total_variance,
        expected_shots_tot=expected_shots_total,
        expected_shots_target=expected_shots_target,
        prob_over=prob_over,
        target_line=target_line,
        time_decay=time_decay,
        bet_result=bet_result,
        odds=args.odds_corners,
    )

    print(report)
    send_telegram_alert(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
