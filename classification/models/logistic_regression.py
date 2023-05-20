from sklearn.model_selection import train_test_split,GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from ..metrics import Metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna.visualization as vis
from joblib import dump, load
import numpy as np
import optuna
optuna.logging.disable_default_handler()

## Logistic Regression
class LR(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self,X,y,params=None):
        if params == None:
            log = LogisticRegression()
            log.fit(X,y)
            self.model = log
        else:
            log = LogisticRegression(**params)
            log.fit(X,y)
            self.model = log
            self.parameters = params

    def create_grid(self, X, y, params=None, cv=3):
        params_columns = ["fit_intercept", "intercept_scaling", 
                          "penalty","C","random_state","solver","max_iter"]
        params_basic = {
                'fit_intercept': [True],
                'intercept_scaling': [0.1,3],
                'penalty': ['l2'],
                'C': [0.1, 1, 10],
                'random_state': [42],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'max_iter': [300]}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
            
        log = LogisticRegression()
        grid_search = GridSearchCV(log, params, cv=cv)
        grid_search.fit(X,y)
        self.model = grid_search.best_estimator_
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y, params=None, n_trials=3, show_plot=False, show_features=False):
        params_columns = ["fit_intercept", "intercept_scaling", 
                          "penalty","C","random_state","solver","max_iter"]
        params_basic = {
                'fit_intercept': [True],
                'intercept_scaling': [0.1,3],
                'penalty': ['l2'],
                'C': [0.1,10],
                'random_state': 42,
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'max_iter': [300, 700]}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        def objective(trial):
            param = {
                'fit_intercept': trial.suggest_categorical("fit_intercept", params['fit_intercept']),
                'intercept_scaling': trial.suggest_float('intercept_scaling', params['intercept_scaling'][0], params['intercept_scaling'][1]),
                'penalty': trial.suggest_categorical("penalty", params['penalty']),
                'C': trial.suggest_float('C', params['C'][0], params['C'][1]),
                'random_state': trial.suggest_int("random_state", params['random_state'], params['random_state']),
                'solver': trial.suggest_categorical("solver", params['solver']),
                'max_iter': trial.suggest_int('max_iter',params['max_iter'][0], params['max_iter'][1])
            }
            lr = LogisticRegression(**param)
            lr.fit(X_test,y_test)
            preds = lr.predict(X_test)
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
        lr_best = LogisticRegression(**best_params)
        lr_best.fit(X, y)
        self.logistic_regression = lr_best
        self.parameters = best_params

    def score(self, X, y):
        preds = np.round(self.model.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)

    def get(self):
        return self.model
    
    def get_parameters(self):
        return self.parameters
    
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
    
    def save(self, model_path="logistic_reg.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)