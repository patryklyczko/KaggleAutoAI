from sklearn.model_selection import GridSearchCV, train_test_split,KFold
from sklearn.metrics import mean_squared_error,r2_score
from ..metrics_regression import Metrics
import optuna.visualization as vis
from catboost import CatBoostRegressor
import numpy as np
import optuna
optuna.logging.disable_default_handler()


class Cat(Metrics):
    def __init__(self):
        self.model = None
        self.parameters = None

    def put(self, model):
        self.model = model

    def create(self, X, y, params=None):
        if params == None:
            gbm = CatBoostRegressor()
            gbm.fit(X,y)
            self.model = gbm
        else:
            gbm = CatBoostRegressor(**params)
            gbm.fit(X,y)
            self.model = gbm
            self.parameters = params

    def create_grid(self, X,y,params=None, cv=3):
        params_columns = ['depth', 'l2_leaf_reg','border_count','feature_border_type'
                          'bagging_temperature','random_strength','fold_permutation_type'
                          'grow_policy','leaf_estimation_method','random_seed', 'verbose']
        params_basic = {
            'depth': [3,13],
            'l2_leaf_reg': [1, 3],
            'border_count': [32,128],
            'bagging_temperature': [0, 0.7],
            'random_strength': [0, 0.7],
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            'leaf_estimation_method': ['Newton', 'Gradient', 'Exact'],
            "feature_border_type": ["Uniform","MinEntropy"],
            "fold_permutation_block": [3],
            'random_seed': [42],
            'verbose': [0]
        }
        if params == None:
                params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
        
        regressor = CatBoostRegressor(loss_function='MAE', eval_metric='MAE')
        grid_search = GridSearchCV(regressor, params, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X,y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        self.model = best_model
        self.parameters = best_params

    def create_optuna(self, X, y, params=None, n_trials=5, show_plot=False, show_features=False):
        params_columns = ['depth', 'l2_leaf_reg','border_count','feature_border_type'
                          'bagging_temperature','random_strength','fold_permutation_type'
                          'grow_policy','leaf_estimation_method','random_seed','verbose']
        params_basic = {
            'depth': [1,5],
            'l2_leaf_reg': [1, 3],
            'border_count': [4,128],
            'bagging_temperature': [0, 0.7],
            'random_strength': [0, 0.7],
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            'leaf_estimation_method': ['Newton', 'Gradient', 'Exact'],
            "feature_border_type": ["Uniform","MinEntropy"],
            "fold_permutation_block": 3,
            'random_seed': 42,
            'verbose': 0
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
                  'depth': trial.suggest_int('depth', params['depth'][0], params['depth'][1]),
                  'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', params['l2_leaf_reg'][0], params['l2_leaf_reg'][1]),
                  'border_count': trial.suggest_int('border_count', params['border_count'][0], params['border_count'][1]),
                  'bagging_temperature': trial.suggest_float('bagging_temperature', params['bagging_temperature'][0],params['bagging_temperature'][1]),
                  'random_strength': trial.suggest_float('random_strength', params['random_strength'][0], params['random_strength'][1]),
                  'grow_policy': trial.suggest_categorical("grow_policy", params['grow_policy']),
                  'leaf_estimation_method': trial.suggest_categorical("leaf_estimation_method", params['leaf_estimation_method']),
                  'feature_border_type': trial.suggest_categorical("feature_border_type", params['feature_border_type']),
                  'fold_permutation_block': trial.suggest_int("fold_permutation_block", params['fold_permutation_block'], params['fold_permutation_block']),
                  'random_seed': trial.suggest_int("random_state", params['random_seed'], params['random_seed']),
                  'verbose': trial.suggest_int("verbose", params['verbose'], params['verbose'])
            }
            gbm = CatBoostRegressor(**param)
            gbm.fit(X_train, y_train)
            y_pred = gbm.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            return mse
        
        study = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()

        best_params = study.best_trial.params
        model = CatBoostRegressor(**best_params)
        self.model = model.fit(X, y)
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
    
    def save(self, model_path="cat.cbm"):
        self.model.save_model(model_path)

    def load(self, model_path):
        self.model.load_model(model_path)