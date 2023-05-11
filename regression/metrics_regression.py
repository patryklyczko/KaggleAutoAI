from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import numpy as np

class Metrics():
  def __init__(self):
    self.mae = 0
    self.mse = 0
    self.rmse = 0 
    self.r2 = 0


  def calculate_metrics(self, y_true, y_preds):
    self.mae = mean_absolute_error(y_true, y_preds)
    self.mse= mean_squared_error(y_true, y_preds)
    self.rmse = np.sqrt(self.mse)
    self.r2 = r2_score(y_true, y_preds)

    metrics = {"mae": self.mae,
               "mse": self.mse,
               "rmse": self.rmse,
               "r2": self.r2}
    
    return metrics