class ValueEngine:
    @staticmethod
    def evaluate_bet(true_probability: float, market_odds: float, bankroll_fraction: float = 0.25) -> dict:
        # Validación de cuotas realistas
        if market_odds <= 1.0:
            return {"error": "Las cuotas de mercado deben ser mayores a 1.0"}
            
        implied_prob = 1.0 / market_odds
        ev = (true_probability * market_odds) - 1.0
        
        # Exige mínimo 3% de valor
        has_value = ev > 0.03  
        
        # Criterio de Kelly
        b = market_odds - 1.0
        p = true_probability
        q = 1.0 - p
        kelly_full = (p * b - q) / b
        
        # Aplicación de Kelly Fraccionado
        recommended_stake_pct = max(0.0, kelly_full * bankroll_fraction)
        
        # Convierte porcentaje a unidades de banca (Ej: 1% de la banca = 1 Unidad)
        stake_units = min(3.0, round(recommended_stake_pct * 100, 1))

        return {
            "ev_percentage": round(ev * 100, 2),
            "implied_prob": round(implied_prob * 100, 2),
            "true_prob": round(true_probability * 100, 2),
            "has_value": has_value,
            # Si no hay valor, la apuesta es 0
            "recommended_stake": stake_units if has_value else 0.0  
        }
