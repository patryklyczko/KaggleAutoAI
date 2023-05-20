from sklearn.model_selection import GridSearchCV, train_test_split,KFold
from sklearn.metrics import mean_squared_error,r2_score
from ..metrics_regression import Metrics
import optuna.visualization as vis
import lightgbm as lgb
import numpy as np
import optuna
optuna.logging.disable_default_handler()


class LGB(Metrics):
    def __init__(self):
        self.model = None
        self.parameters = None
    
    def put(self, model):
        self.model = model

    def create(self, X, y, params=None):
        if params == None:
            gbm = lgb.LGBMRegressor()
            gbm.fit(X,y)
            self.model = gbm
        else:
            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(X,y)
            self.model = gbm
            self.parameters = params

    def create_grid(self, X, y, params=None, cv=3):
        params_columns = ['boosting_type', 'objective', 'num_leaves',
                          'n_estimators', 'max_depth', 'min_child_samples',
                          'subsample', 'colsample_bytree', 'seed', 'max_bin']
        params_basic = {
            'boosting_type': ['gbdt'],
            'objective': ['regression'],
            'num_leaves': [20,200],
            'n_estimators': [30, 100],
            'max_depth': [5, 32],
            'min_child_samples': [30,70],
            'subsample': [0.3, 0.9],
            'colsample_bytree': [0.5, 1.0],
            'seed': [42],
            'max_bin': [255]
        }
        if params == None:
                params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        gbm = lgb.LGBMRegressor()
        grid_search = GridSearchCV(gbm, params, cv=cv, n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        self.model = best_model
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y, params=None, n_trials=5, show_plot=False, show_features=False):
        params_columns = ['boosting_type', 'objective', 'num_leaves',
                          'n_estimators', 'max_depth', 'min_child_samples',
                          'subsample', 'colsample_bytree', 'seed', 'max_bin']
        params_basic = {
            'boosting_type': ['gbdt'],
            'objective': 'regression',
            'num_leaves': [10,100],
            'n_estimators': [10, 200],
            'max_depth': [1, 6],
            'min_child_samples': [10,70],
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'seed': 42,
            'max_bin': 255
        }
        if params == None:
                params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        def objective(trial):
            param = {
                'boosting_type': trial.suggest_categorical("boosting_type", params['boosting_type']),
                'objective': trial.suggest_categorical("objective", params['objective']),
                'num_leaves': trial.suggest_int('num_leaves', params['num_leaves'][0], params['num_leaves'][1]),
                'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1]),
                'max_depth': trial.suggest_int("max_depth", params['max_depth'][0], params['max_depth'][1]),
                'min_child_samples': trial.suggest_int("min_child_samples", params['min_child_samples'][0], params['min_child_samples'][1]),
                'subsample': trial.suggest_float("subsample", params['subsample']),
                'colsample_bytree': trial.suggest_float("colsamlpe_bytree", params['colsample_bytree']),
                'seed': trial.suggest_int("seed", params['seed'], params['seed']),
                'max_bin': trial.suggest_int("max_bin", params['max_bin'], params['max_bin'])
            }
            gbm = lgb.LGBMRegressor(**param)
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
        gbm_best = lgb.LGBMRegressor(**best_params)
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
    
    def save(self, model_path='lgb.txt'):
        self.model.save_model(model_path)

    def load(self, model_path):
        self.model =lgb.Booster(model_file=model_path)