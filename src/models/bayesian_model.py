class BayesianHierarchicalEngine:
    @staticmethod
    def posterior_corners_estimate(home_feat: dict, away_feat: dict, league_avg: float = 9.5) -> float:
        """Calcula la media posterior combinando priors de liga y rendimiento de equipos."""
        prior_mean = league_avg
        prior_var = 2.0
        
        # Las medias esperadas se combinan de forma aditiva
        obs_mean = home_feat["weighted_corners"] + away_feat["weighted_corners"]
        
        # Ley de variables independientes: La varianza de la suma es la suma de las varianzas
        obs_var = max(1.0, home_feat["var_corners"] + away_feat["var_corners"])
        
        # Ponderación Bayesiana por precisiones estadísticas
        precision_prior = 1.0 / prior_var
        precision_obs = 1.0 / obs_var
        
        posterior_mean = ((prior_mean * precision_prior) + (obs_mean * precision_obs)) / (precision_prior + precision_obs)
        return float(posterior_mean)
