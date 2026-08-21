import argparse
from src.api_client import StatsAPIClient
from src.feature_engineering import extract_advanced_features
from src.models.glm_poisson import GLMCountModel
from src.models.bayesian_model import BayesianHierarchicalEngine
from src.value_finder import ValueEngine
from src.telegram_bot import send_telegram_alert

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Se esperaba un valor booleano.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--match-id", default="")
    parser.add_argument("--odds-corners", type=float, default=0.0)
    parser.add_argument("--live", type=str2bool, default=False)
    parser.add_argument("--minute", type=int, default=0)
    args = parser.parse_args()

    client = StatsAPIClient()

    # 1. Resolución de Identidades mediante búsqueda paramétrica corregida
    home_id = client.search_team_id(args.home)
    away_id = client.search_team_id(args.away)

    # 2. Extracción recursiva asíncrona de datos históricos
    h_matches = client.get_last_10_matches_stats(home_id)
    a_matches = client.get_last_10_matches_stats(away_id)

    h_feat = extract_advanced_features(h_matches, home_id)
    a_feat = extract_advanced_features(a_matches, away_id)

    # 3. Modelado Estadístico
    bayesian_corners = BayesianHierarchicalEngine.posterior_corners_estimate(h_feat, a_feat)
    expected_shots_tot = h_feat["weighted_shots_total"] + a_feat["weighted_shots_total"]
    expected_shots_target = h_feat["weighted_shots_target"] + a_feat["weighted_shots_target"]
    total_variance = h_feat["var_corners"] + a_feat["var_corners"]

    time_decay = 1.0
    target_line = 9.5

    # 4. Modificaciones condicionales de tiempo In-Play
    if args.live and args.minute > 0:
        time_decay = max(0.0, (90 - args.minute) / 90.0)
        bayesian_corners *= time_decay
        total_variance *= time_decay

    prob_over_9_5_c = GLMCountModel.predict_negative_binomial(
        lambda_exp=bayesian_corners,
        variance=total_variance,
        line=target_line
    )

    # 5. Análisis Matemático de Cuotas de Mercado
    bet_res = {"has_value": False, "ev_percentage": 0.0, "recommended_stake": 0.0}
    if args.odds_corners > 1.0:
        bet_res = ValueEngine.evaluate_bet(prob_over_9_5_c, args.odds_corners)

    # 6. Salida de Datos y Reporte Automatizado
    live_status = f"🔴 EN VIVO (Minuto {args.minute})" if args.live else "📅 PRE-PARTIDO"
    msg = (
        f"🤖 *QUANT ENGINE V2 REPORT* | {live_status}\n"
        f"⚽ *{args.home.upper()} vs {args.away.upper()}*\n\n"
        f"📊 *MÉTRICAS PREDICTIVAS (Ponderadas):*\n"
        f"• xT Local: `{h_feat.get('xT', 0.0):.2f}` | Visitante: `{a_feat.get('xT', 0.0):.2f}`\n"
        f"• RoT (Precisión Tiro): `{h_feat.get('RoT', 0.0)*100:.1f}%` / `{a_feat.get('RoT', 0.0)*100:.1f}%`\n"
        f"• PPDA (Presión): `{h_feat.get('PPDA', 12.0):.1f}` / `{a_feat.get('PPDA', 12.0):.1f}`\n"
        f"• CII (Juego Bandas): `{h_feat.get('CII', 0.0):.1f}` / `{a_feat.get('CII', 0.0):.1f}`\n\n"
        f"🎯 *PROYECCIONES RESTANTES EN EL ENCUENTRO:*\n"
        f"• Córneres Esperados: `{bayesian_corners:.2f}` (Var: {total_variance:.2f})\n"
        f"• Remates Totales: `{expected_shots_tot * time_decay:.2f}`\n"
        f"• Remates al Arco: `{expected_shots_target * time_decay:.2f}`\n"
        f"📈 *Probabilidad Over {target_line}:* `{prob_over_9_5_c * 100:.2f}%`\n"
    )

    if args.odds_corners > 1.0:
        msg += (
            f"\n💰 *EVALUACIÓN DE CUOTA ({args.odds_corners})*\n"
            f"• EV+: `{bet_res['ev_percentage']}%`\n"
            f"• ¿Hay Valor?: `{'SÍ' if bet_res['has_value'] else 'NO'}`\n"
            f"• Stake Sugerido (Kelly): `{bet_res['recommended_stake']} / 3`\n"
        )

    print(msg)
    send_telegram_alert(msg)

if __name__ == "__main__":
    main()
