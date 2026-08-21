import argparse
import sys
from src.api_client import StatsAPIClient
from src.feature_engineering import extract_advanced_features
from src.models.glm_poisson import GLMCountModel
from src.models.bayesian_model import BayesianHierarchicalEngine
from src.value_finder import ValueEngine
from src.telegram_bot import send_telegram_alert

def str2bool(v):
    """Función auxiliar para parsear booleanos correctamente desde strings de la CLI."""
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
    # CORRECCIÓN 1: Usar la función helper para evitar que "false" se evalúe como True
    parser.add_argument("--live", type=str2bool, default=False)
    parser.add_argument("--minute", type=int, default=0)
    args = parser.parse_args()

    client = StatsAPIClient()

    # 1. Búsqueda de IDs
    home_id = client.search_team_id(args.home)
    away_id = client.search_team_id(args.away)

    # 2. Extracción de estadísticas de últimos 10 partidos
    h_matches = client.get_last_10_matches_stats(home_id)
    a_matches = client.get_last_10_matches_stats(away_id)

    h_feat = extract_advanced_features(h_matches, home_id)
    a_feat = extract_advanced_features(a_matches, away_id)

    # 3. Modelado Base (Pre-partido de 90 minutos)
    bayesian_corners = BayesianHierarchicalEngine.posterior_corners_estimate(h_feat, a_feat)
    
    # CORRECCIÓN 2: El volumen de remates esperados en el partido es la suma de ambos, no una multiplicación por xG
    expected_shots_tot = h_feat["weighted_shots_total"] + a_feat["weighted_shots_total"]
    expected_shots_target = (h_feat["weighted_shots_target"] + a_feat["weighted_shots_target"])

    # CORRECCIÓN 3: Suma de varianzas para el GLM Count Model
    total_variance = h_feat["var_corners"] + a_feat["var_corners"]

    # 4. Ajuste por Dinámica In-Play (Si el partido está corriendo)
    current_corners_in_play = 0
    target_line = 9.5

    if args.live and args.minute > 0:
        # Extraer córneres actuales si contamos con el match_id en vivo
        if args.match_id:
            # Aquí podrías consultar un endpoint live. Como fallback o simulación, asumimos capturar la línea restante.
            pass
        
        # Proporción de tiempo restante en el partido
        time_decay = max(0.0, (90 - args.minute) / 90.0)
        
        # Escalamos la media y la varianza para el tiempo restante (Proceso de Poisson / NB)
        bayesian_corners = bayesian_corners * time_decay
        total_variance = total_variance * time_decay
        
        # Si ya conocemos los córneres actuales del partido en vivo, la línea a superar se encoge:
        # target_line = max(0.5, 9.5 - current_corners_in_play)

    # Ejecución del Modelo probabilístico discreto
    prob_over_9_5_c = GLMCountModel.predict_negative_binomial(
        lambda_exp=bayesian_corners,
        variance=total_variance,
        line=target_line
    )

    # 5. Evaluación de Valor
    bet_res = {"has_value": False, "ev_percentage": 0.0, "recommended_stake": 0.0}
    if args.odds_corners > 1.0:
        bet_res = ValueEngine.evaluate_bet(prob_over_9_5_c, args.odds_corners)

    # 6. Formateo y Salida
    live_status = f"🔴 EN VIVO (Minuto {args.minute})" if args.live else "📅 PRE-PARTIDO"
    
    msg = (
        f"🤖 *QUANT ENGINE V2 REPORT* | {live_status}\n"
        f"⚽ *{args.home.upper()} vs {args.away.upper()}*\n\n"
        f"📊 *MÉTRICAS PREDICTIVAS (Ponderadas):*\n"
        f"• xT Local: `{h_feat['xT']:.2f}` | Visitante: `{a_feat['xT']:.2f}`\n"
        f"• RoT (Precisión Tiro): `{h_feat['RoT']*100:.1f}%` / `{a_feat['RoT']*100:.1f}%`\n"
        f"• PPDA (Presión): `{h_feat['PPDA']:.1f}` / `{a_feat['PPDA']:.1f}`\n"
        f"• CII (Juego Bandas): `{h_feat['CII']:.1f}` / `{a_feat['CII']:.1f}`\n\n"
        f"🎯 *PROYECCIONES RESTANTES EN EL ENCUENTRO:*\n"
        f"• Córneres Esperados: `{bayesian_corners:.2f}` (Var: {total_variance:.2f})\n"
        f"• Remates Totales: `{expected_shots_tot * (time_decay if args.live else 1.0):.2f}`\n"
        f"• Remates al Arco: `{expected_shots_target * (time_decay if args.live else 1.0):.2f}`\n"
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
