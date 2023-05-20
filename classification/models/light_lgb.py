from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import optuna.integration.lightgbm as lgb
from ..metrics import Metrics
import optuna.visualization as vis
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import optuna
optuna.logging.disable_default_handler()


class LightLGB(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self, X, y, params=None):
        if params == None:
            cat = lgb(objective="binary")
            cat.fit(X,y)
            self.model = cat
        else:
            cat = lgb(**params)
            cat.fit(X,y)
            self.model = cat
            self.parameters = params

    def create_grid(self,X,y,params=None,cv=3,available_memory_gb=5):
        rows = len(X)
        memory_row = X.memory_usage(deep=True).sum() / rows
        max_bin = int(np.floor(available_memory_gb * 1024**3 / (rows * memory_row)))

        params_columns = ["max_bin","objective","metric","boosting_type",
                          "lambda_l1","lambda_l2","num_leaves","feature_fraction",
                          "bagging_fraction","bagging_freq","min_child_samples",
                          "max_depth","min_gain_to_split","seed","num_boost_round",
                          "verbosity"]
        params_basic ={
            "max_bin": [max_bin],
            "objective": ["binary"],
            "metric": ["binary_error"],
            "boosting_type": ['gbdt', 'dart'],
            "lambda_l1": [1e-4, 1],
            "lambda_l2": [1e-4,1],
            "num_leaves": [3, 100],
            "feature_fraction": [0.2, 0.8],
            "bagging_freq": [0.1, 0.9],
            "min_child_samples": [2,100],
            "max_depth": [1, 7],
            "min_gain_to_split": [2, 30],
            "seed": [42],
            "num_boost_round": [100],
            "verbosity": [False]
        }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        lgb = lgb()
        grid_search = GridSearchCV(lgb, params, cv=cv)
        grid_search.fit(X, y)
        model = grid_search.best_estimator_
        self.model = model
        self.parameters = grid_search.best_params_

    def create_optuna(self, X, y, params=None, n_trials=10,available_memory_gb=5, show_plot=False, show_features=False):
        ## Calculate processor usage 
        rows = len(X)
        memory_row = X.memory_usage(deep=True).sum() / rows
        max_bin = int(np.floor(available_memory_gb * 1024**3 / (rows * memory_row)))

        params_columns = ["max_bin","objective","metric","boosting_type",
                          "lambda_l1","lambda_l2","num_leaves","feature_fraction",
                          "bagging_freq","min_child_samples","num_boost_round",
                          "max_depth","min_gain_to_split","seed",
                          "verbosity"]
        params_basic ={
            "max_bin": max_bin,
            "objective": ["binary"],
            "metric": "binary_error",
            "boosting_type": ['gbdt', 'dart'],
            "lambda_l1": [0, 1],
            "lambda_l2": [0,1],
            "num_leaves": [3, 100],
            "feature_fraction": [0.2, 0.8],
            "bagging_freq": [0.1, 0.9],
            "min_child_samples": [2,100],
            "max_depth": [1, 7],
            "min_gain_to_split": [2, 30],
            "seed": 42,
            "num_boost_round": 100,
            "verbosity": False
        }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        def objective(trial):
            param = {
                'objective': trial.suggest_categorical('objective', params["objective"]),
                'metric': trial.suggest_categorical('metric', params["metric"]),
                'max_bin': trial.suggest_int("max_bin", params["max_bin"], params["max_bin"]),
                'boosting_type': trial.suggest_categorical("boosting_type", params["boosting_type"]),
                'lambda_l1': trial.suggest_float('lambda_l1', params["lambda_l1"][0], params["lambda_1"][1]),
                'lambda_l2': trial.suggest_float('lambda_l2', params["lambda_l2"][0], params["lambda_2"][1]),
                'num_leaves': trial.suggest_int('num_leaves', params["num_leaves"][0], params["num_leaves"][1]),
                'feature_fraction': trial.suggest_uniform('feature_fraction', params["feature_fraction"][0], params["feature_fraction"][1]),
                'bagging_freq': trial.suggest_int('bagging_freq', params["bagging_freq"][0], params["bagging_freq"][1]),
                'min_child_samples': trial.suggest_int('min_child_samples', params["min_child_samples"][0], params["min_child_samples"][0]),
                'max_depth': trial.suggest_int('max_depth', params["max_depth"][0], params["max_depth"][1]),
                'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', params["min_gain_to_split"][0], params["min_gain_to_split"][1]),
                'seed': trial.suggest_int("seed", params["seed"], params["seed"]),
                'num_boost_round': trial.suggest_int("num_boost_round", params["num_boost_round"], params["num_boost_round"]),
                'verbosity': params["verbosity"]
            }
            gbm = lgb.train(param, dtrain,valid_sets=dval)
            preds = gbm.predict(X_train)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(y_train, pred_labels)
            return accuracy

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()
            
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        best_params = study.best_trial.params
        booster = lgb.train(best_params, dtrain, valid_sets=dval,
                            verbose_eval=0)
        self.model=booster
        self.parameters = best_params

    def score(self,X,y):
        preds = self.model.predict(X)
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
    
    def save(self, model_path='lgb.txt'):
        self.model.save_model(model_path)

    def load(self, model_path):
        self.model =lgb.Booster(model_file=model_path)