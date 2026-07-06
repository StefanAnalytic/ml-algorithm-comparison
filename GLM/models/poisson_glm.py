import statsmodels.api as sm

class PoissonGLM:
    """Unser starres, aber mathematisch sauberes Zähldaten-Modell."""
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train):
        # Statsmodels braucht eine Konstante für den Intercept
        X_train_const = sm.add_constant(X_train.values)
        glm = sm.GLM(y_train, X_train_const, family=sm.families.Poisson(link=sm.families.links.Log()))
        self.model = glm.fit(cov_type='HC3')
        
    def predict(self, X_test):
        X_test_const = sm.add_constant(X_test.values)
        return self.model.predict(X_test_const)