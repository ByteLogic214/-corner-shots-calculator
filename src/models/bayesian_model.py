class BayesianHierarchicalEngine:
    @staticmethod
    def posterior_corners_estimate(home_feat: dict, away_feat: dict, league_avg: float = 9.5) -> float:
        """Ajuste Bayesiano conjugado Normal-Normal aplicando aditividad de varianzas independientes."""
        prior_mean = league_avg
        prior_var = 2.0
        
        obs_mean = home_feat.get("weighted_corners", 4.5) + away_feat.get("weighted_corners", 4.5)
        obs_var = max(1.0, home_feat.get("var_corners", 1.0) + away_feat.get("var_corners", 1.0))
        
        precision_prior = 1.0 / prior_var
        precision_obs = 1.0 / obs_var
        
        posterior_mean = ((prior_mean * precision_prior) + (obs_mean * precision_obs)) / (precision_prior + precision_obs)
        return float(posterior_mean)
