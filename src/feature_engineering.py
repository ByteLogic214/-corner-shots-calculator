import numpy as np

def extract_advanced_features(matches: list, team_id: int) -> dict:
    if not matches:
        return {}

    xt_list, rot_list, cii_list, ppda_list = [], [], [], []
    xg_list, xa_list = [], []
    corners_list, shots_tot_list, shots_target_list = [], [], []

    for match in matches:
        st = match.get("detailed_stats", {})
        # Determinar herencia de claves del equipo de forma local o externa en el payload
        is_home = (match.get("home_team_id") == team_id) or (match.get("home_team", {}).get("id") == team_id)
        pfx = "home" if is_home else "away"

        # Extracción y fallback según el esquema de objetos anidados de la API
        side_data = st.get(pfx, st)
        
        c = side_data.get("corners" if "corners" in side_data else f"corners_{pfx}", 0)
        st_tot = side_data.get("shots" if "shots" in side_data else f"total_shots_{pfx}", 0)
        st_tar = side_data.get("shots_on_target" if "shots_on_target" in side_data else f"shots_on_target_{pfx}", 0)
        xg = side_data.get("xg" if "xg" in side_data else f"xg_{pfx}", 1.0)
        xa = side_data.get("xa" if "xa" in side_data else f"xa_{pfx}", 1.0)
        
        blocked_crosses = side_data.get("blocked_crosses" if "blocked_crosses" in side_data else f"blocked_crosses_{pfx}", 0)
        wing_duels = side_data.get("wing_duels_won" if "wing_duels_won" in side_data else f"wing_duels_won_{pfx}", 0)
        ppda = side_data.get("ppda" if "ppda" in side_data else f"ppda_{pfx}", 12.0)
        xt = side_data.get("expected_threat" if "expected_threat" in side_data else f"expected_threat_{pfx}", 1.0)

        corners_list.append(c)
        shots_tot_list.append(st_tot)
        shots_target_list.append(st_tar)
        xg_list.append(xg)
        xa_list.append(xa)
        
        xt_list.append(xt)
        rot_list.append(st_tar / st_tot if st_tot > 0 else 0.33)
        cii_list.append(blocked_crosses + wing_duels)
        ppda_list.append(ppda)

    n_matches = len(matches)
    time_indices = np.arange(n_matches)[::-1]
    
    weights = np.exp(-0.1 * time_indices)
    weights /= weights.sum()

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
