from sklearn.model_selection import GridSearchCV, train_test_split,KFold
from sklearn.metrics import r2_score
from ..metrics_regression import Metrics
from sklearn.linear_model import ElasticNet
import numpy as np
from joblib import dump, load

class ElasticNetM(Metrics):
    def __init__(self):
        self.model = None
        self.parameters = None

    def put(self, model):
        self.model = model

    def create(self, X, y, params=None):
        if params == None:
            gbm = ElasticNet()
            gbm.fit(X,y)
            self.model = gbm
        else:
            gbm = ElasticNet(**params)
            gbm.fit(X,y)
            self.model = gbm
            self.parameters = params
    
    def create_grid(self,X,y, params=None, cv=3):
        params_columns = ['alpha','l1_ratio', 'max_iter','random_state']

        params_basic = {
            'alpha': [0.01, 0.1, 1, 5, 10],
            'l1_ratio': [0.2, 0.5, 0.8],
            'max_iter': [1000, 5000, 7000],
            'random_state': [42]
        }
        if params == None:
                params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
        
        elastic_net = ElasticNet()
        grid_search = GridSearchCV(elastic_net, params, cv=cv)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        self.model = best_model
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
    
    def save(self, model_path="elastic_net.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)