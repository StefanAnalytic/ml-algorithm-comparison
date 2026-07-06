import statsmodels.api as sm

class NegativeBinomialGLM:
    """Das Negative Binomial GLM: Besser bei extremer Streuung (Overdispersion) als Poisson."""
    def __init__(self, alpha=1.0):
        # alpha ist der Dispersionsparameter. Je höher, desto mehr Streuung erlaubt das Modell.
        self.alpha = alpha 
        self.model = None

    def fit(self, X_train, y_train):
        X_train_const = sm.add_constant(X_train.values)
        glm = sm.GLM(
            y_train, 
            X_train_const, 
            family=sm.families.NegativeBinomial(alpha=self.alpha, link=sm.families.links.Log())
        )
        self.model = glm.fit(cov_type='HC3')

    def predict(self, X_test):
        X_test_const = sm.add_constant(X_test.values)
        return self.model.predict(X_test_const)