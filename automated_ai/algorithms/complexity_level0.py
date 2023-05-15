from ...model_data.model_data import ModelData
from sklearn.model_selection import train_test_split, KFold


def Complexity_Level0(model, X, y):
    print(f"Phase I \t Phase II \t Phase III \t Phase IV\n")
    importance_columns = {}
    for col in X.columns:
        importance_columns[col] = 0

    ## fix null columns
    model_data = ModelData(X)
    columns_null = model_data.find_null_column()
    for col in columns_null:
        column_data_type = X[col].dtype
        if column_data_type == "object":
            model_data.fix_column_str(col)
        else:
            model_data.fix_columns([col], "mean")
    print(f"Completed", end=" ")
    
    ## Text column to dummies
    for col in X.columns:
        if X[col].dtype == "object" and X[col].nunique() < 10:
            model_data.dummies(col)
    X = model_data.return_dataframe()

    ## Train model in diffrent samples to extract importance of columns
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        model.create_optuna(X_train,y_train,n_trials=50)
        feature_model = model.get()
        features = feature_model.get_feature_importance()
        for i,col in enumerate(X.columns):
            importance_columns[col] += features[i]
    print("-------Complete", end=" ")
    
    ## Acceptting column that satify the requirenments 
    mean_importance = sum(importance_columns.values()) / len(importance_columns)
    threshold_importance = 0.75 * mean_importance
    columns = []
    for k,v in importance_columns.items():
        if v > threshold_importance:
            columns.append(k)
    print("-------Complete", end=" ")

    X = X[columns]
    model.create_optuna(X,y,n_trials=300)
    print("-------Complete", end=" ")
    return model.get(), columns

