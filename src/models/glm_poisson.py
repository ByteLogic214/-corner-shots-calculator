from scipy.stats import nbinom, poisson

class GLMCountModel:
    @staticmethod
    def predict_negative_binomial(lambda_exp: float, variance: float, line: float) -> float:
        """Ajuste discreto de conteo con corrección de continuidad determinista."""
        if lambda_exp <= 0:
            return 0.0
            
        epsilon = 1e-4 
        int_floor = int(line) 
        
        if variance <= (lambda_exp + epsilon):
            return float(1.0 - poisson.cdf(int_floor, lambda_exp))
        
        r = (lambda_exp ** 2) / (variance - lambda_exp)
        p = lambda_exp / variance
        
        prob_over = 1.0 - nbinom.cdf(int_floor, r, p)
        return float(prob_over)
