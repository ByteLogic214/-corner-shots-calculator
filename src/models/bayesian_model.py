"""Bayesian hierarchical model for corner kick estimation."""

from typing import Optional


class BayesianHierarchicalEngine:
    """Conjugate Normal-Normal Bayesian updater for corner expectations."""

    DEFAULT_LEAGUE_AVG: float = 9.5
    DEFAULT_PRIOR_VAR: float = 2.0
    DEFAULT_MIN_OBS_VAR: float = 1.0

    @staticmethod
    def posterior_corners_estimate(
        home_feat: dict[str, float],
        away_feat: dict[str, float],
        league_avg: Optional[float] = None,
    ) -> float:
        """Compute posterior expected corners via Normal-Normal conjugacy.

        Args:
            home_feat: Weighted features for the home team.
            away_feat: Weighted features for the away team.
            league_avg: Prior mean for total corners in the league.

        Returns:
            Posterior mean estimate of total corners.
        """
        if league_avg is None:
            league_avg = BayesianHierarchicalEngine.DEFAULT_LEAGUE_AVG

        prior_mean = float(league_avg)
        prior_var = BayesianHierarchicalEngine.DEFAULT_PRIOR_VAR

        home_corners = float(home_feat.get("weighted_corners", 4.5))
        away_corners = float(away_feat.get("weighted_corners", 4.5))
        obs_mean = home_corners + away_corners

        home_var = float(home_feat.get("var_corners", 1.0))
        away_var = float(away_feat.get("var_corners", 1.0))
        obs_var = max(
            BayesianHierarchicalEngine.DEFAULT_MIN_OBS_VAR,
            home_var + away_var,
        )

        precision_prior = 1.0 / prior_var
        precision_obs = 1.0 / obs_var

        posterior_mean = (
            (prior_mean * precision_prior) + (obs_mean * precision_obs)
        ) / (precision_prior + precision_obs)

        return float(posterior_mean)
