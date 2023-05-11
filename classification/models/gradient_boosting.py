from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier
from ..metrics import Metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import optuna


class GradientBoosting(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create_grid(self, X, y, params=None, cv=2):
        params_columns = ["loss","learning_rate", "n_estimators",
                          "subsample","min_samples_split","max_depth",
                          "random_state","max_features","verbose",
                          "validation_fraction"]
        params_basic = {
                'loss': ['log_loss'],
                'learning_rate': [0.001,0.01,0.1],
                'n_estimators': [10, 100],
                'subsample': [0.8],
                'min_samples_split': [3,10],
                'max_depth': [3, 10],
                'random_state': [42],
                'max_features': ['sqrt', None],
                'verbose': [0],
                'validation_fraction': [0.2] 
            }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
            
        grad = GradientBoostingClassifier()
        grid_search = GridSearchCV(grad,params,cv=cv)
        grid_search.fit(X,y)
        self.model = grid_search.best_estimator_
        self.parameters = grid_search.best_params_


    def create_gb_optuna(self, X, y, params=None, n_trials=3):
        params_columns = ["loss","learning_rate", "n_estimators",
                          "subsample","min_samples_split","max_depth",
                          "random_state","max_features","verbose",
                          "validation_fraction"]
        params_basic = {
                'loss': ['log_loss'],
                'learning_rate': [0.001,0.1],
                'n_estimators': [10, 500],
                'subsample': [0.8],
                'min_samples_split': [3,10],
                'max_depth': [1, 8],
                'random_state': [42],
                'max_features': ['sqrt'],
                'verbose': [0],
                'validation_fraction': [0.2] 
            }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        def objective(trail):
            param = {
                'loss': params['loss'],
                'learning_rate': trail.suggest_loguniform("learning_rate", params['learning_rate'][0], params['learning_rate'][1]),
                'n_estimators': trail.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1]),
                'subsample': params['subsample'],
                'min_samples_split': trail.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1]),
                'max_depth': trail.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1]),
                'random_state': params["random_state"],
                'max_features': params['max_features'],
                'verbose': 0,
                'validation_fraction': 0.2
            }
            gb = GradientBoostingClassifier(**param)
            gb.fit(X_train, y_train)
            preds = gb.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        gb_best = GradientBoostingClassifier(**best_params, random_state=42)
        gb_best.fit(X, y)
        self.model= gb_best
        self.parameters = best_params


    def score(self, X, y):
        preds = self.model.predict(X)
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_kfold(self, X, y, df_test, n_splits=5, params=None):
        if params == None:
            params = self.parameters
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(df_test.shape[0])
        roc = []
        n=0

        for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
            X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_test = y.iloc[train_index], y[valid_index]

            self.create(X_train,y_train,params=params)
            predictions += self.predict(df_test)/n_splits
            val_pred = self.predict(X_test)
            roc.append(roc_auc_score(y_test,val_pred))

            print(f"{i} Fold scored: {roc[i]}")

        print(f"Mean roc score {np.mean(roc)}")
        return predictions

    def get(self):
        return self.model
    
    def get_parameters(self):
        return self.parameters