from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from ..metrics import Metrics
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import xgboost as xgb
import optuna


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

    def create_optuna(self,X,y,params=None,n_trials=3):
        params_columns = ['n_estimators','learning_rate','max_depth',
                          'min_child_weight', 'subsample','colsample_bytree',
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
            'objective': 'binary:logistic', 
            'eval_metric': 'logloss'}
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
                'subsample': params['subsample'],
                'colsample_bytree': trial.suggest_float('colsample_bytree', params['colsample_bytree'][0], params['colsample_bytree'][1]),
                'gamma': trial.suggest_float('gamma', params['gamma'][0], params['gamma'][1]),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', params['scale_pos_weight'][0], params['scale_pos_weight'][1]),
                'objective': params['objective'],
                'random_state': 42,
                'eval_metric': params['eval_metric']
            }
            mxgb = xgb.XGBClassifier(**param)
            mxgb.fit(X_train, y_train)
            preds = mxgb.predict(X_test)
            logloss = log_loss(y_test, preds)
            return logloss
        study = optuna.create_study(direction='minimize')
        study.optimize(objective,n_trials=n_trials)
        best_params = study.best_params
        xgb_best = xgb.XGBClassifier(**best_params, random_state=42)
        xgb_best.fit(X, y)
        self.model = xgb_best
        self.parameters = best_params

    def score(self, X, y):
        preds = np.round(self.xgb.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.xgb.predict(X)
    
    def evaluate_kfold(self, X, y, df_test, n_splits=5, params=None):
        if params == None:
            params = self.parameters
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = np.zeros(df_test.shape[0])
        roc = []
        n=0

        for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
            X_train, X_test = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_test = y.iloc[train_index], y.iloc[valid_index]

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
    