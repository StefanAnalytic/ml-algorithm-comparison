from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    """Entscheidungsbäume, die im Wald zusammenarbeiten. Robust, aber keine Extrapolation."""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1 # Nutzt alle CPU-Kerne deines Macs
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)