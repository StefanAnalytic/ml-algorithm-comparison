import lightgbm as lgb

class LightGBMModel:
    """Der pfeilschnelle Herausforderer von Microsoft. Oft schneller als XGBoost."""
    def __init__(self, random_state=42):
        # Auch hier: Zähldaten-Optimierung durch Poisson!
        self.model = lgb.LGBMRegressor(
            objective='poisson',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1 # Unterdrückt unwichtige Warnungen im Terminal
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)