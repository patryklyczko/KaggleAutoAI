from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split,KFold
import optuna.visualization as vis
from ..metrics_regression import Metrics
from joblib import dump, load
import numpy as np
import optuna
optuna.logging.disable_default_handler()


class GradientBoosting(Metrics):
    def __init__(self):
        self.model = None
        self.parameters = None

    def put(self, model):
        self.model = model

    def create(self, X, y, params=None):
        if params == None:
          gbm = GradientBoostingRegressor()
          gbm.fit(X,y)
          self.model = gbm
        else:
            gbm = GradientBoostingRegressor(**params)
            gbm.fit(X,y)
            self.model = gbm
            self.parameters = params
          
    def create_grid(self, X, y,params=None, scoring='neg_mean_squared_error', n_jobs =-1, cv=5):
        params_columns = ['learning_rate', 'n_estimators', 'max_depth',
                        'min_samples_split', 'min_samples_leaf', 'max_features',
                        'subsample', 'loss', 'alpha', 'random_state', 'validation_fraction']
        params_basic = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [10, 200],
            'max_depth': [1,10],
            'min_samples_split': [10,30],
            'min_samples_leaf': [10,100],
            'max_features': [0.5, 'sqrt', 10],
            'subsample': [0.5, 1],
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'alpha': [0.1, 0.9],
            'random_state': [42],
            'validation_fraction': [0.2]
            }
        if params == None:
                params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        gbm = GradientBoostingRegressor()
        grid_search = GridSearchCV(gbm, params, cv=cv, scoring=scoring,n_jobs=n_jobs)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        self.model = best_model
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y, params=None, n_trials=5, show_plot=False, show_features=False):
        params_columns = ['learning_rate', 'n_estimators', 'max_depth',
                        'min_samples_split', 'min_samples_leaf', 'max_features',
                        'subsample', 'loss', 'alpha', 'random_state', 'validation_fraction']
        params_basic = {
            'learning_rate': [0.001, 0.2],
            'n_estimators': [10, 300],
            'max_depth': [1,10],
            'min_samples_split': [10,50],
            'min_samples_leaf': [10,100],
            'max_features': ['sqrt'],
            'subsample': 0.8,
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'alpha': [0.1, 0.9],
            'random_state': 42,
            'validation_fraction': 0.2
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
                'learning_rate': trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][0]),
                'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical("max_features", params['max_features']),
                'subsample': trial.suggest_float("subsample", params['subsample'], params['subsample']),
                'loss': trial.suggest_categorical("loss", params['loss']),
                'alpha': trial.suggest_float('alpha', params['alpha'][0], params['alpha'][1]),
                'random_state': trial.suggest_int("random_state", params['random_state'], params['random_state']),
                'validation_fraction': trial.suggest_float("validation_fraction", params['validation_fraction'], params['validation_fraction'])
            }
            gbm = GradientBoostingRegressor(**param)
            gbm.fit(X_train, y_train)
            y_pred = gbm.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            return mse
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()

        best_params = study.best_params
        gbm_best = GradientBoostingRegressor(**best_params)
        gbm_best.fit(X, y)
        self.model= gbm_best
        self.parameters = best_params
    
    def score(self, X, y):
        preds = np.round(self.model.predict(X))
        return self.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_kfold(self, X, y, df_test, n_splits=5, params=None):
        if params == None:
            params = self.parameters
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(shape=(df_test.shape[0],))
        r2 = []
        n=0

        for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
            X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_test = y.iloc[train_index], y.iloc[valid_index]
            self.create(X_train,y_train,params=params)
            predictions += self.predict(df_test)/n_splits
            val_pred = self.predict(X_test)
            r2.append(r2_score(y_test,val_pred))

            print(f"{i} Fold scored: {r2[i]}")

        print(f"Mean r2_score {np.mean(r2)}")
        return predictions

    def get(self):
        return self.model
    
    def get_parameters(self):
        return self.parameters

    def save(self, model_path="gradient_boosting.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)


