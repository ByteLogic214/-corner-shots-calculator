import numpy as np

def _safe_numeric(value, sub_key="total", default=0) -> float:
    """
    Función auxiliar recursiva para desempaquetar diccionarios profundamente anidados
    y extraer de forma segura valores numéricos reales, evitando colapsos por tipo de dato.
    """
    if value is None:
        return float(default)
        
    # Si sigue siendo un diccionario, se descompone de forma iterativa profunda
    while isinstance(value, dict):
        if not value: # Diccionario vacío
            return float(default)
        # Intenta extraer mediante las claves estándar ordenadas por prioridad
        next_value = value.get(sub_key, value.get("value", value.get("total", value.get("total_shots", None))))
        if next_value is None:
            # Fallback: Si no encuentra las claves, toma el primer valor del diccionario
            first_key = list(value.keys())[0]
            value = value[first_key]
        else:
            value = next_value

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def extract_advanced_features(matches: list, team_id: int) -> dict:
    if not matches:
        return {}

    xt_list, rot_list, cii_list, ppda_list = [], [], [], []
    xg_list, xa_list = [], []
    corners_list, shots_tot_list, shots_target_list = [], [], []

    for match in matches:
        st = match.get("detailed_stats", {})
        if not st:
            continue
            
        is_home = (match.get("home_team_id") == team_id) or (match.get("home_team", {}).get("id") == team_id)
        pfx = "home" if is_home else "away"

        # Extraer el bloque de datos correspondiente al lado del equipo
        side_data = st.get(pfx, st)

        # 1. Extracción de métricas base
        raw_c = side_data.get("corners", side_data.get(f"corners_{pfx}", 0))
        raw_st_tot = side_data.get("shots", side_data.get("total_shots", side_data.get(f"total_shots_{pfx}", 0)))
        raw_st_tar = side_data.get("shots_on_target", side_data.get(f"shots_on_target_{pfx}", 0))
        raw_xg = side_data.get("xg", side_data.get(f"xg_{pfx}", 1.0))
        raw_xa = side_data.get("xa", side_data.get(f"xa_{pfx}", 1.0))
        
        raw_blocked_crosses = side_data.get("blocked_crosses", side_data.get(f"blocked_crosses_{pfx}", 0))
        raw_wing_duels = side_data.get("wing_duels_won", side_data.get(f"wing_duels_won_{pfx}", 0))
        raw_ppda = side_data.get("ppda", side_data.get(f"ppda_{pfx}", 12.0))
        raw_xt = side_data.get("expected_threat", side_data.get(f"expected_threat_{pfx}", 1.0))

        # 2. Normalización profunda robusta anti-anidamiento
        c = _safe_numeric(raw_c, sub_key="total", default=0)
        st_tot = _safe_numeric(raw_st_tot, sub_key="total", default=0)
        st_tar = _safe_numeric(raw_st_tar, sub_key="total", default=0)
        xg = _safe_numeric(raw_xg, sub_key="total", default=1.0)
        xa = _safe_numeric(raw_xa, sub_key="total", default=1.0)
        
        blocked_crosses = _safe_numeric(raw_blocked_crosses, sub_key="total", default=0)
        wing_duels = _safe_numeric(raw_wing_duels, sub_key="total", default=0)
        ppda = _safe_numeric(raw_ppda, sub_key="value", default=12.0)
        xt = _safe_numeric(raw_xt, sub_key="total", default=1.0)

        # 3. Almacenamiento y cálculo de ratios tácticos continuos
        corners_list.append(c)
        shots_tot_list.append(st_tot)
        shots_target_list.append(st_tar)
        xg_list.append(xg)
        xa_list.append(xa)
        
        xt_list.append(xt)
        rot_list.append(st_tar / st_tot if st_tot > 0 else 0.33)
        cii_list.append(blocked_crosses + wing_duels)
        ppda_list.append(ppda)

    if not corners_list:
        return {}

    # 4. Generación de pesos con decaimiento exponencial temporal
    n_matches = len(corners_list)
    time_indices = np.arange(n_matches)[::-1]
    
    weights = np.exp(-0.1 * time_indices)
    weights /= weights.sum()

    # 5. Reducción matemática ponderada limpia
    weighted_corners_mean = np.average(corners_list, weights=weights)
    variance_corners = np.average((np.array(corners_list) - weighted_corners_mean)**2, weights=weights)

    return {
        "weighted_corners": float(weighted_corners_mean),
        "weighted_shots_total": float(np.average(shots_tot_list, weights=weights)),
        "weighted_shots_target": float(np.average(shots_target_list, weights=weights)),
        "xT": float(np.average(xt_list, weights=weights)),
        "RoT": float(np.average(rot_list, weights=weights)),
        "CII": float(np.average(cii_list, weights=weights)),
        "PPDA": float(np.average(ppda_list, weights=weights)),
        "xG": float(np.average(xg_list, weights=weights)),
        "xA": float(np.average(xa_list, weights=weights)),
        "var_corners": float(variance_corners)
    }
