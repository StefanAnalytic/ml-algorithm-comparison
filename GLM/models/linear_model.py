from sklearn.linear_model import Ridge

class LinearBaseline:
    """Die einfache Baseline: Eine lineare Regression (Ridge)."""
    def __init__(self):
        # Wir nutzen Ridge (Lineare Regression mit leichter Regularisierung)
        self.model = Ridge(alpha=1.0)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)