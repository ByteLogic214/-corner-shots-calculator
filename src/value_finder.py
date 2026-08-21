class ValueEngine:
    @staticmethod
    def evaluate_bet(true_probability: float, market_odds: float, bankroll_fraction: float = 0.25) -> dict:
        """Cálculo fraccionado de Kelly con umbral mínimo de valor y techo de unidades."""
        if market_odds <= 1.0:
            return {"error": "Las cuotas de mercado deben ser mayores a 1.0"}
            
        implied_prob = 1.0 / market_odds
        ev = (true_probability * market_odds) - 1.0
        has_value = ev > 0.03  
        
        b = market_odds - 1.0
        p = true_probability
        q = 1.0 - p
        kelly_full = (p * b - q) / b if b > 0 else 0
        
        recommended_stake_pct = max(0.0, kelly_full * bankroll_fraction)
        stake_units = min(3.0, round(recommended_stake_pct * 100, 1))

        return {
            "ev_percentage": round(ev * 100, 2),
            "implied_prob": round(implied_prob * 100, 2),
            "true_prob": round(true_probability * 100, 2),
            "has_value": has_value,
            "recommended_stake": stake_units if has_value else 0.0
        }
