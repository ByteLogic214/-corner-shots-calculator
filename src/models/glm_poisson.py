"""Generalised Linear Model (GLM) count predictions."""

from scipy.stats import nbinom, poisson


class GLMCountModel:
    """Count-based over/under probability engine."""

    EPSILON: float = 1e-4

    @staticmethod
    def predict_negative_binomial(
        lambda_exp: float,
        variance: float,
        line: float,
    ) -> float:
        """Predict P(X > line) using a Negative-Binomial or Poisson model.

        If the variance is close to the mean (var <= mean + eps), a Poisson
        approximation is used. Otherwise, the method of moments is applied
        to fit a Negative-Binomial distribution.

        Args:
            lambda_exp: Expected count (mean).
            variance: Observed variance (must be > 0).
            line: The over/under threshold (e.g. 9.5).

        Returns:
            Probability that the count exceeds the line.
        """
        if lambda_exp <= 0.0:
            return 0.0

        if variance <= 0.0:
            variance = lambda_exp + GLMCountModel.EPSILON

        int_floor = int(line)

        # Poisson limit: variance ≈ mean
        if variance <= (lambda_exp + GLMCountModel.EPSILON):
            return float(1.0 - poisson.cdf(int_floor, lambda_exp))

        # Negative-Binomial method of moments
        # Var = mu + mu^2 / r  =>  r = mu^2 / (var - mu)
        # p = mu / var
        try:
            r = (lambda_exp ** 2) / (variance - lambda_exp)
            p = lambda_exp / variance
        except ZeroDivisionError:
            return float(1.0 - poisson.cdf(int_floor, lambda_exp))

        prob_over = 1.0 - nbinom.cdf(int_floor, r, p)
        return float(prob_over)
