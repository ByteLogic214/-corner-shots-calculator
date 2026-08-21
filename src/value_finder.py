"""Betting value and stake recommendation engine."""

from typing import Optional


class ValueEngine:
    """Kelly criterion and expected value calculator."""

    DEFAULT_BANKROLL_FRACTION: float = 0.25
    EV_THRESHOLD: float = 0.03
    MAX_STAKE_UNITS: float = 3.0

    @staticmethod
    def evaluate_bet(
        true_probability: float,
        market_odds: float,
        bankroll_fraction: Optional[float] = None,
    ) -> dict[str, float | bool]:
        """Evaluate a bet using fractional Kelly criterion.

        Args:
            true_probability: Model-estimated probability of success (0-1).
            market_odds: Decimal odds offered by the market (> 1.0).
            bankroll_fraction: Fraction of full Kelly to use (default 0.25).

        Returns:
            Dictionary with keys:
                - ev_percentage: Expected value as a percentage.
                - implied_prob: Market-implied probability as a percentage.
                - true_prob: Provided true probability as a percentage.
                - has_value: True if EV exceeds the 3 % threshold.
                - recommended_stake: Suggested stake in units (capped at 3.0),
                  or 0.0 if no value exists.
        """
        if bankroll_fraction is None:
            bankroll_fraction = ValueEngine.DEFAULT_BANKROLL_FRACTION

        if market_odds <= 1.0:
            raise ValueError("Market odds must be strictly greater than 1.0")

        if not (0.0 <= true_probability <= 1.0):
            raise ValueError("true_probability must be in the range [0, 1]")

        implied_prob = 1.0 / market_odds
        ev = (true_probability * market_odds) - 1.0
        has_value = ev > ValueEngine.EV_THRESHOLD

        b = market_odds - 1.0  # net odds received
        p = true_probability
        q = 1.0 - p

        kelly_full = (p * b - q) / b if b > 0 else 0.0
        recommended_stake_pct = max(0.0, kelly_full * bankroll_fraction)
        stake_units = min(
            ValueEngine.MAX_STAKE_UNITS,
            round(recommended_stake_pct * 100, 1),
        )

        return {
            "ev_percentage": round(ev * 100, 2),
            "implied_prob": round(implied_prob * 100, 2),
            "true_prob": round(true_probability * 100, 2),
            "has_value": has_value,
            "recommended_stake": stake_units if has_value else 0.0,
        }
