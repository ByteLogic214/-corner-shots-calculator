from scipy.stats import nbinom, poisson

class GLMCountModel:
    @staticmethod
    def predict_negative_binomial(lambda_exp: float, variance: float, line: float) -> float:
        """
        Calcula la probabilidad de que el conteo supere una línea de apuesta dada.
        Ajusta por sobredispersión usando Binomial Negativa si la varianza > media.
        """
        # Validaciones de seguridad para evitar errores numéricos
        if lambda_exp <= 0:
            return 0.0
            
        # Tolerancia pequeña para estabilidad numérica (evitar r gigante)
        epsilon = 1e-4 
        
        # Caso 1: Sin sobredispersión (Distribución de Poisson)
        if variance <= (lambda_exp + epsilon):
            # Para variables discretas, 'Más de X' requiere evaluar el entero inferior
            # Si line = 9.5 -> int_floor = 9 -> cdf(9) es Prob(<=9) -> 1 - cdf es Prob(>=10)
            int_floor = int(line) 
            return float(1.0 - poisson.cdf(int_floor, lambda_exp))
        
        # Caso 2: Con sobredispersión (Distribución Binomial Negativa)
        # Parametrización estándar basada en momentos (Media y Varianza)
        r = (lambda_exp ** 2) / (variance - lambda_exp)
        p = lambda_exp / variance
        
        int_floor = int(line)
        prob_over = 1.0 - nbinom.cdf(int_floor, r, p)
        return float(prob_over)
