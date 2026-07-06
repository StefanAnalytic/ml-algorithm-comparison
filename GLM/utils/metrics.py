import numpy as np
from sklearn.metrics import mean_poisson_deviance, root_mean_squared_error, mean_absolute_error

def evaluate_model(model_name, y_true, y_pred):
    # Vor negativen Werten schützen (für Poisson Deviance zwingend nötig)
    y_pred_safe = np.clip(y_pred, a_min=1e-5, a_max=None)
    
    metrics = {
        "Model": model_name,
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Poisson_Dev": mean_poisson_deviance(y_true, y_pred_safe)
    }
    return metrics