from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import ExtraTreesClassifier
from ..metrics import Metrics
import optuna.visualization as vis
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump, load
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import OneHotEncoder
optuna.logging.disable_default_handler()


class ExtraTrees(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None
        self.targets = None

    def create(self,X,y,params):
      if params == None:
        rf = ExtraTreesClassifier()
        rf.fit(X,y)
        self.model = rf
      else:
        rf = ExtraTreesClassifier(**params)
        rf.fit(X,y)
        self.model = rf
        self.parameters=params

    def create_grid(self, X,y, params=None, cv=2):
        params_columns = ["n_estimators","criterion","max_depth",
                          "min_samples_split","min_samples_leaf",
                          "min_weight_fraction_leaf","n_jobs",
                          "random_state"]
        params_basic = {
                'n_estimators': [10, 100],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [7, 15, None],
                'min_samples_split': [2, 20],
                'min_samples_leaf': [1, 10],
                'min_weight_fraction_leaf': [0, 0.5],
                'n_jobs': [-1],
                'random_state': [42]}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
            
        rf = ExtraTreesClassifier()
        grid_search = GridSearchCV(rf, params, cv=cv)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y,params=None,n_trials=2,show_plot=False, show_features=False):
        params_columns = ["n_estimators","criterion","max_depth",
                          "min_samples_split","min_samples_leaf",
                          "min_weight_fraction_leaf","n_jobs",
                          "random_state"]
        params_basic = {
                'n_estimators': [10, 100],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [1,5],
                'min_samples_split': [2, 20],
                'min_samples_leaf': [1, 10],
                'min_weight_fraction_leaf': [0, 0.5],
                'n_jobs': -1,
                'random_state': 42}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        def objective(trial):
            param = {
                'criterion': trial.suggest_categorical("criterion", params['criterion']),
                'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1]),
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', params['min_weight_fraction_leaf'][0], params['min_weight_fraction_leaf'][1]),
                'n_jobs': trial.suggest_int("n_jobs", params["n_jobs"], params["n_jobs"]),
                'random_state': trial.suggest_int("random_state", params["random_state"], params["random_state"])
            }
            trees = ExtraTreesClassifier(**param)
            trees.fit(X_train,y_train)
            preds = trees.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            return accuracy

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()

        best_params = study.best_params
        trees_best = ExtraTreesClassifier(**best_params)
        trees_best.fit(X_train, y_train)
        self.model = trees_best
        self.parameters = best_params

    def score(self, X, y):
        preds = np.round(self.model.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_kfold(self, X, y, df_test, n_splits=5, params=None, classes=1):
        if params == None:
            params = self.parameters
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(shape=(df_test.shape[0],classes))
        roc = []
        n=0

        for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
            X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_test = y.iloc[train_index], y.iloc[valid_index]
            self.create(X_train,y_train,params=params)
            predictions += np.expand_dims(self.predict(df_test)/n_splits, axis=-1)

            val_pred = self.predict(X_test)
            if classes > 1:
              roc.append(roc_auc_score(y_test,val_pred,multi_class='ovr'))
            else:
              roc.append(roc_auc_score(y_test,val_pred))
            print(f"{i} Fold scored: {roc[i]}")

        print(f"Mean roc score {np.mean(roc)}")
        return predictions

    def get(self):
        return self.model
    
    def get_parameters(self):
        return self.parameters

    def set_targets(self,y):
      enc = OneHotEncoder(handle_unknown='ignore')
      y = enc.fit_transform(y.values.reshape(-1,1))
      targets = {key: value for key,value in enumerate(enc.categories_[0])}
      self.targets = targets
      return pd.DataFrame(y.toarray())

    def label_target(self,y):
      return [self.targets[np.argmax(i)] for i in y]

    def get_target(self):
      return self.targets

    def label2number(self, y):
      target_number = {}
      values = []
      for k,v in self.targets.items():
        target_number[v] = k
      for i in y:
        values.append(target_number[i])
      return values
    
    def save(self, model_path="extra_trees.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)
