from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from ..metrics import Metrics
import optuna.visualization as vis
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import pandas as pd
import numpy as np
import optuna
optuna.logging.disable_default_handler()

class Cat(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self, X, y, params=None):
        if params == None:
            cat = CatBoostClassifier()
            cat.fit(X,y)
            self.model = cat
        else:
            cat = CatBoostClassifier(**params)
            cat.fit(X,y)
            self.model = cat
            self.parameters = params

    def create_grid(self,X, y, params=None,cv=3, ram_limit=8):
        ram_limit *= 1024*1024*1024
        params_columns = ["depth","l2_leaf_reg","random_strength",
                          "thread_count","bootstrap_type","scale_pos_weight",
                          "user_ram_limit","verbose"]
        
        params_basic = {
            'depth': [3,7,10],
            'l2_leaf_reg': [1,10],
            'random_strength': [0.1,0.9],
            'thread_count': [-1],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
            'scale_pos_weight': [1, 3],
            'used_ram_limit': [ram_limit],
            'num_boost_round': 100,
            'verbose': [0],
            'random_state': [42]}
        
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        cat = CatBoostClassifier()
        grid_search = GridSearchCV(cat, params, cv=cv)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.parameters = best_params

    def create_optuna(self,X,y,params=None,n_trials=5,ram_limit=5, show_plot=False, show_features=False):
        ram_limit *= 1024*1024*1024
        params_columns = ["depth","l2_leaf_reg","random_strength",
                          "thread_count","bootstrap_type","scale_pos_weight",
                          "used_ram_limit","verbose","random_state",'num_boost_round']
        
        params_basic = {
            'depth': [1,8],
            'l2_leaf_reg': [0,10],
            'random_strength': [0.01,0.9],
            'thread_count': -1,
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
            'scale_pos_weight': [0, 5],
            'num_boost_round': 200,
            'used_ram_limit': ram_limit,
            'verbose': 0,
            'random_state': 42}
        
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        def objective_binary(trial):
            param = {
                "depth": trial.suggest_int("depth", params["depth"][0], params["depth"][1]),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", params["l2_leaf_reg"][0], params["l2_leaf_reg"][1]),
                "random_strength": trial.suggest_float("random_strength", params["random_strength"][0], params["random_strength"][1]),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", params["scale_pos_weight"][0], params["scale_pos_weight"][1]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", params["bootstrap_type"]),
                "used_ram_limit": trial.suggest_int("used_ram_limit", params["used_ram_limit"], params["used_ram_limit"]),
                "verbose": trial.suggest_int("verbose", params["verbose"], params["verbose"]),
                "thread_count": trial.suggest_int("thread_count", params["thread_count"], params["thread_count"]),
                "random_state": trial.suggest_int("random_state", params['random_state'], params['random_state']),
                'num_boost_round': trial.suggest_int("num_boost_round", params['num_boost_round'], params['num_boost_round'])
            }
            cat = CatBoostClassifier(**param)
            cat.fit(X_train,y_train)
            preds = cat.predict(X_test)
            loss = log_loss(y_test, preds)
            return loss

        def objective_multi(trial):
          param = {
                "depth": trial.suggest_int("depth", params["depth"][0], params["depth"][1]),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", params["l2_leaf_reg"][0], params["l2_leaf_reg"][1]),
                "random_strength": trial.suggest_float("random_strength", params["random_strength"][0], params["random_strength"][1]),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", params["scale_pos_weight"][0], params["scale_pos_weight"][1]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", params["bootstrap_type"]),
                "used_ram_limit": trial.suggest_int("used_ram_limit", params["used_ram_limit"], params["used_ram_limit"]),
                "verbose": trial.suggest_int("verbose", params["verbose"], params["verbose"]),
                "thread_count": trial.suggest_int("thread_count", params["thread_count"], params["thread_count"]),
                "random_state": trial.suggest_int("random_state", params['random_state'], params['random_state']),
                'num_boost_round': trial.suggest_int("num_boost_round", params['num_boost_round'], params['num_boost_round'])
            }
          cat = CatBoostClassifier(**param,loss_function='MultiClass')
          cat.fit(X_train,y_train)
          preds = cat.predict_proba(X_test)
          loss = log_loss(y_test, preds)
          return loss


        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        if max(y) > 1:
          study.optimize(objective_multi, n_trials=n_trials)
          best_params = study.best_params
          cat = CatBoostClassifier(**best_params,loss_function='MultiClass')
        else:
          study.optimize(objective_binary, n_trials=n_trials)
          best_params = study.best_params
          cat = CatBoostClassifier(**best_params)

        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()

        cat.fit(X, y)
        self.model = cat
        self.parameters = best_params


    def score(self,X,y):
        preds = np.round(self.model.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_kfold(self, X, y, df_test, n_splits=5, params=None,classes=1):
        X, y = pd.DataFrame(X), pd.DataFrame(y)
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
    
    def save(self, model_path="cat.cbm"):
        self.model.save_model(model_path)

    def load(self, model_path):
        self.model.load_model(model_path)