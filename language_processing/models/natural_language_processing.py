from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

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
  

class LanguageProcessing():
  def __init__(self) -> None:
    self.bayes = MultinomialNB()

  def create_baseline_raw(self, X_raw, y_raw):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])

    params = {
        'vect__max_features': [100, 1000],
        'vect__strip_accents': ['ascii'],
        'vect__min_df': np.linspace(0.0001,0.3, 3),
        'vect__ngram_range': [(1, 1), (1, 2),(1,3),(2,1)],
        'clf__alpha': np.linspace(0.001,1,10)
    }
    grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
    grid_search.fit(X_raw, y_raw)
    print(f"Best parameters: {grid_search.best_params_}")
    self.bayes = grid_search.best_estimator_


  def create_baseline(self, X, y, params_alpha):
    bayes = MultinomialNB()
    params = {
        'alpha': params_alpha
    }
    grid_search = GridSearchCV(bayes, params, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    self.bayes = grid_search.best_estimator_

    

