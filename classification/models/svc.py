from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from ..metrics import Metrics
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np


class SVCModel(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self, X, y, params=None):
        if params == None:
            cat = SVC()
            cat.fit(X,y)
            self.model = cat
        else:
            cat = SVC(**params)
            cat.fit(X,y)
            self.model = cat
            self.parameters = params

    def create_grid(self,X,y,params=None, cv=3, cache_size=100):
        params_columns = ["C", "kernel", "gamma", "cache_size"]
        params_basic = {'C': [0.1,1,10],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'cache_size': [cache_size]}
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        svc = SVC()
        grid_search = GridSearchCV(svc, params, cv=cv)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.parameters = best_params

    def score(self,X,y):
        preds = np.round(self.model.predict(X))
        return self.metric.calculate_metrics(y, preds)

    def predict(self, X):
        return self.model.predict(X)

    def get(self):
        return self.model
    
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
    
    def get_parameters(self):
        return self.parameters
    
    def save(self, model_path="svc.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)