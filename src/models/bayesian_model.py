class BayesianHierarchicalEngine:
    @staticmethod
    def posterior_corners_estimate(home_feat: dict, away_feat: dict, league_avg: float = 9.5) -> float:
        """
        Estimación Bayesiana posterior de la media de córneres totales en un partido.
        Aplica un modelo conjugado Normal-Normal para regularizar muestras pequeñas.
        """
        # Prior de la liga (conocimiento previo global)
        prior_mean = league_avg
        prior_var = 2.0
        
        # Evidencia Observada (Suma de las medias de ambos equipos)
        obs_mean = home_feat["weighted_corners"] + away_feat["weighted_corners"]
        
        # CORRECCIÓN: La varianza de la suma de dos variables independientes es la SUMA de sus varianzas.
        # Mantener el piso de seguridad (ej. 1.0 para evitar sobreajuste en muestras chicas)
        obs_var = max(1.0, home_feat["var_corners"] + away_feat["var_corners"])
        
        # Actualización Bayesiana (Media ponderada por la precisión [1/varianza])
        precision_prior = 1.0 / prior_var
        precision_obs = 1.0 / obs_var
        
        posterior_mean = ((prior_mean * precision_prior) + (obs_mean * precision_obs)) / (precision_prior + precision_obs)
        
        return float(posterior_mean)
