import xgboost as xgb

class XGBoostModel:
    """Der Kaggle-Champion. Lernt iterativ aus seinen eigenen Fehlern."""
    def __init__(self, random_state=42):
        # WICHTIG: objective='count:poisson'. Wir zwingen XGBoost, wie ein GLM zu denken!
        self.model = xgb.XGBRegressor(
            objective='count:poisson',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)