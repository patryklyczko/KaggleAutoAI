from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

class Metrics():
  def __init__(self):
    self.accuracy = 0
    self.precision = 0
    self.f1_score = 0 
    self.recall = 0

  def calculate_metrics(self, y_true, y_preds):
    self.accuracy = accuracy_score(y_true, y_preds)
    self.f1_score = f1_score(y_true, y_preds)
    self.recall = recall_score(y_true, y_preds)
    self.precision = precision_score(y_true, y_preds)

    metrics = {"accuracy": self.accuracy,
               "f1_score": self.f1_score,
               "recall": self.recall,
               "precision": self.precision}
    
    return metrics