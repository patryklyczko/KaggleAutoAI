from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from ..metrics import Metrics
import optuna.visualization as vis
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import xgboost as xgb
import optuna
optuna.logging.disable_default_handler()


class XGB(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self,X,y,params=None,cv=3):
        if params == None:
            params={
                "n_estimators": 50,
            }
        xgb = xgb.XGBClassifier(**params)
        xgb.fit(X,y)
        self.model = xgb
        self.parameters = params

    def create_grid(self, X,y, params=None, cv=3):
        params_columns = ['n_estimators','learning_rate','max_depth',
                          'min_child_weight', 'subsample','colsample_bytree',
                          'gamma','scale_pos_weight','objective','eval_metric']
        params_basic = {
                'n_estimators': [100, 500],  
                'learning_rate': [0.01, 0.1], 
                'max_depth': [1, 7],  
                'min_child_weight': [1, 10],  
                'subsample': [0.8],  
                'colsample_bytree': [0.6, 1.0],  
                'gamma': [0, 0.3],  
                'scale_pos_weight': [1,10], 
                'objective': ['binary:logistic'], 
                'eval_metric': ['logloss', 'auc'],
                'random_state': [42]}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        mxgb = xgb.XGBClassifier()
        grid_search = GridSearchCV(mxgb,params,cv=cv)
        grid_search.fit(X,y)
        model = grid_search.best_estimator_
        self.model = model
        self.parameters = grid_search.best_params_

    def create_optuna(self,X,y,params=None,n_trials=3, show_plot=False, show_features=False):
        params_columns = ['n_estimators','learning_rate','max_depth', 'random_state',
                          'min_child_weight', 'subsample','colsample_bytree', 'tree_method',
                          'gamma','scale_pos_weight','objective','eval_metric']
        params_basic = {
            'n_estimators': [100, 500],  
            'learning_rate': [0.01, 0.1], 
            'max_depth': [1, 7],  
            'min_child_weight': [1, 10],  
            'subsample': 0.8,  
            'colsample_bytree': [0.6, 1.0],  
            'gamma': [0, 0.3],  
            'scale_pos_weight': [1,10], 
            'objective': ['binary:logistic'], 
            'eval_metric': ['logloss'],
            'tree_method': ['approx'],
            "random_state": 42}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1]),
                'learning_rate': trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1]),
                'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1]),
                'min_child_weight': trial.suggest_int('min_child_weight', params['min_child_weight'][0], params['min_child_weight'][1]),
                'subsample': trial.suggest_float("subsample", params['subsample'], params['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', params['colsample_bytree'][0], params['colsample_bytree'][1]),
                'gamma': trial.suggest_float('gamma', params['gamma'][0], params['gamma'][1]),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', params['scale_pos_weight'][0], params['scale_pos_weight'][1]),
                'objective': trial.suggest_categorical("objective", params['objective']),
                'tree_method': trial.suggest_categorical("tree_method", params['tree_method']),
                'random_state': trial.suggest_int("random_state", params["random_state"], params["random_state"]),
                'eval_metric': trial.suggest_categorical("eval_metric", params['eval_metric'])
            }
            mxgb = xgb.XGBClassifier(**param)
            mxgb.fit(X_train, y_train)
            preds = mxgb.predict(X_test)
            logloss = log_loss(y_test, preds)
            return logloss
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective,n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()

        best_params = study.best_params
        xgb_best = xgb.XGBClassifier(**best_params)
        xgb_best.fit(X, y)
        self.model = xgb_best
        self.parameters = best_params

    def score(self, X, y):
        preds = np.round(self.xgb.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.xgb.predict(X)
    
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
    
    def save(self, model_path='xgb.bin'):
        self.model.save_model(model_path)

    def load(self, model_path):
        self.model.load_model(model_path)
    