from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from ..metrics import Metrics
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
import numpy as np


class SGD(Metrics):
    def __init__(self):
        self.metric = Metrics()
        self.model = None
        self.parameters = None

    def create(self, X, y, params=None, cv=3):
        params_columns = ["loss", "penalty","alpha","max_iter"]
        params_basic = {
            'loss': ['hinge', 'log_loss', 'modified_huber','perceptron'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.001, 0.01, 0.1,0.5],
            'max_iter': [1000, 5000]
            }
        if params == None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]

        sgd = SGDClassifier()
        grid_search = GridSearchCV(sgd, params, cv=cv)
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
            y_train, y_test = y.iloc[train_index], y.iloc[valid_index]

            self.create(X_train,y_train,params=params)
            predictions += self.predict(df_test)/n_splits
            val_pred = self.predict(X_test)
            roc.append(roc_auc_score(y_test,val_pred))

            print(f"{i} Fold scored: {roc[i]}")

        print(f"Mean roc score {np.mean(roc)}")
        return predictions
    
    def get_parameters(self):
        return self.parameters
    
    def save(self, model_path="sgd.joblib"):
        dump(self.model, model_path)
    
    def load(self, model_path):
        self.model = load(model_path)