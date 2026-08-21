from scipy.stats import nbinom, poisson

class GLMCountModel:
    @staticmethod
    def predict_negative_binomial(lambda_exp: float, variance: float, line: float) -> float:
        """Calcula la probabilidad exacta de superar una línea discreta de conteo."""
        if lambda_exp <= 0:
            return 0.0
            
        epsilon = 1e-4 
        int_floor = int(line) 
        
        # Caso A: Ausencia de sobredispersión estructural (Colapso analítico a Poisson)
        if variance <= (lambda_exp + epsilon):
            return float(1.0 - poisson.cdf(int_floor, lambda_exp))
        
        # Caso B: Sobredispersión presente (Ajuste por momentos en Binomial Negativa)
        r = (lambda_exp ** 2) / (variance - lambda_exp)
        p = lambda_exp / variance
        
        prob_over = 1.0 - nbinom.cdf(int_floor, r, p)
        return float(prob_over)
