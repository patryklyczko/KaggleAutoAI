from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import optuna.integration.lightgbm as lgb
from ..metrics import Metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import optuna

class LightLGB(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

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

    def create_lgb_optuna(self, X, y, params, n_trials,available_memory_gb, boosting_type):
        ## Calculate processor usage 
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        def objective(trial):
            param = {
                'objective': params["objective"],
                'metric': params["metric"],
                'max_bin': params["max_bin"],
                'boosting_type': param["boosting_type"],
                'lambda_l1': trial.suggest_loguniform('lambda_l1', params["lambda_l1"][0], params["lambda_1"][1]),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', params["lambda_l2"][0], params["lambda_2"][1]),
                'num_leaves': trial.suggest_int('num_leaves', params["num_leaves"][0], params["num_leaves"][1]),
                'feature_fraction': trial.suggest_uniform('feature_fraction', params["feature_fraction"][0], params["feature_fraction"][1]),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', params["bagging_freaction"][0], params["bagging_fraction"][1]),
                'bagging_freq': trial.suggest_int('bagging_freq', params["bagging_freq"][0], params["bagging_freq"][1]),
                'min_child_samples': trial.suggest_int('min_child_samples', params["min_child_samples"][0], params["min_child_samples"][0]),
                'max_depth': trial.suggest_int('max_depth', params["max_depth"][0], params["max_depth"][1]),
                'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', params["min_gain_to_split"][0], params["min_gain_to_split"][1]),
                'seed': params["bin"],
                'num_boost_round': params["num_boost_round"],
                'verbosity': params["verbosity"]
            }
            gbm = lgb.train(param, dtrain,valid_sets=dval)
            preds = gbm.predict(X_train)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(y_train, pred_labels)
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        best_params = study.best_trial.params
        best_params['objective'] = params["objective"]
        best_params['metric'] = params['metric']
        best_params['seed'] = params["seed"]
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