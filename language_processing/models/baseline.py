from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from ..data_manipulation.metrics import Metrics
import numpy as np

class Baseline():
  def __init__(self) -> None:
    self.model = None
    self.parameters = None

  def create_raw(self, X_raw, y_raw,params=None, cv=5):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    params_columns = ['vect__max_features','vect__strip_accents','vect__min_df',
                      'vect__ngram_range','clf__alpha']
    
    params_basic = {
        'vect__max_features': [100, 1000],
        'vect__strip_accents': ['ascii'],
        'vect__min_df': np.linspace(0.0001,0.3, 3),
        'vect__ngram_range': [(1, 1), (1, 2),(1,3),(2,1)],
        'clf__alpha': np.linspace(0.001,1,10)
    }
    if params == None:
            params = params_basic
    else:
        for parameter in params_columns:
            if parameter not in params.keys():
                params[parameter] = params_basic[parameter]

    grid_search = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1)
    grid_search.fit(X_raw, y_raw)
    self.parameters = grid_search.best_params_
    self.model = grid_search.best_estimator_


  def create(self, X, y, params, cv=5):
    params_columns = ['alpha']
    
    params_basic = {
        'alpha': np.linspace(0.001,1,10)
    }
    if params == None:
            params = params_basic
    else:
        for parameter in params_columns:
            if parameter not in params.keys():
                params[parameter] = params_basic[parameter]

    bayes = MultinomialNB()
    grid_search = GridSearchCV(bayes, params, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    self.parameters = grid_search.best_params_
    self.model = grid_search.best_estimator_

  def get(self):
    return self.model

    

