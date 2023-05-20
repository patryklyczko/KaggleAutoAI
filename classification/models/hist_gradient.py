from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from ..metrics import Metrics
import optuna.visualization as vis
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump, load
import numpy as np
import optuna
optuna.logging.disable_default_handler()


class HistGradient(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self,X,y,params=None):
        if params == None:
            hist = HistGradientBoostingClassifier()
            hist.fit(X,y)
            self.model = hist
        else:
            hist = HistGradientBoostingClassifier(**params)
            hist.fit(X,y)
            self.model = hist
            self.parameters = params

    def create_grid(self, X, y, params=None, cv=3):
        params_columns = ["loss", "learning_rate", "max_iter",
                          "max_leaf_nodes","max_depth","min_samples_leaf"
                          ,"l2_regularization","max_bins","validation_fraction","verbose",
                          "random_state"]
        params_basic = {
            'loss': ['log_loss'],
            'learning_rate': [0.01, 0.1],
            'max_iter': [200, 1000],
            'max_leaf_nodes': [13, 31],
            'max_depth': [3, 15, None],
            'min_samples_leaf': [10, 30],
            'l2_regularization': [0, 2],
            'max_bins': [255],
            'validation_fraction': [0.2],
            'verbose': [0],
            'random_state': [42]
        }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        hist = HistGradientBoostingClassifier()
        grid_search = GridSearchCV(hist, params, cv=cv)
        grid_search.fit(X,y)
        self.model = grid_search.best_estimator_
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y, params=None, n_trials=2, show_plot=False, show_features=False):
        params_columns = ["loss", "learning_rate", "max_iter",
                            "max_leaf_nodes","max_depth","min_samples_leaf"
                            ,"l2_regularization","max_bins","validation_fraction","verbose",
                            "random_state"]
        params_basic = {
            'loss': ['log_loss'],
            'learning_rate': [0.01, 0.1],
            'max_iter': [200, 1000],
            'max_leaf_nodes': [13, 31],
            'max_depth': [1, 7],
            'min_samples_leaf': [10, 30],
            'l2_regularization': [0, 2],
            'max_bins': 255,
            'validation_fraction': 0.2,
            'verbose': 0,
            'random_state': 42
        }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        def objective(trial):
            param = {
                'loss': trial.suggest_categorical("loss", params['loss']),
                'learning_rate': trial.suggest_loguniform('learning_rate', params['learning_rate'][0], params['learning_rate'][1]),
                'max_iter': trial.suggest_int('max_iter', params['max_iter'][0], params['max_iter'][1]),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', params['max_leaf_nodes'][0], params['max_leaf_nodes'][1]),
                'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1]),
                'l2_regularization': trial.suggest_float('l2_regularization', params["l2_regularization"][0], params["l2_regularization"][1]),
                'max_bins': trial.suggest_int('max_bins', params['max_bins'], params['max_bins']),
                'validation_fraction': trial.suggest_int("validation_fraction", params['validation_fraction'], params['validation_fraction']),
                'verbose': trial.suggest_int("verbose", params['verbose'], params['verbose']),
                'random_state': trial.suggest_int("random_state", params["random_state"], params["random_state"])
            }
            hist = HistGradientBoostingClassifier(**param)
            hist.fit(X_train, y_train)
            preds = hist.predict(X_test)
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
        hist_best = HistGradientBoostingClassifier(**best_params)
        hist_best.fit(X, y)
        self.model = hist_best
        self.parameters = best_params


    def score(self, X, y):
        preds = self.model.predict(X)
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
    
    def save(self, model_path="hist_gradient.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)