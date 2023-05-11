from typing import List
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

class ModelData:
  def __init__(self, df):
    self.df_model = df

  def return_dataframe(self):
    return self.df_model

  def head(self):
    return self.df_model.head()

  def columns(self):
    return self.df_model.columns

  def unique(self, column):
    return self.df_model[column].unique()

  def bar(self, column):
    for col in column:
      self.df_model[column].value_counts().sort_values().plot(kind = "barh")
      plt.show()

  def sample(self,frac):
    return self.df_model.sample(frac=frac)
  
  def info(self):
    return self.df_model.info()
  
  def describe(self):
    return self.df_model.describe()

  def null(self):
    return self.df_model.isnull().sum()

  def find_null_column(self) -> List:
    columns = self.df_model.columns[self.df_model.isnull().any()]
    return columns

  def dropnull(self):
    self.df_model.dropna(inplace=True)

  def column_type(self, columns):
    for col in columns:
      print(f"{col}: {self.df_model[col].dtypes} \n")

  def dtypes_type(self, type):
    return self.df_model.select_dtypes(include=[type]).columns
  
  def rename(self, replace, replace_with):
    self.df_model = self.df_model.rename(columns={replace: replace_with})
  
  def drop(self, column):
    self.df_model = self.df_model.drop(column, axis=1)

  def fix_columns(self, columns:List[str] , method: str):
    if method == "mean":
      for column in columns:
        self.df_model[column].fillna(self.df_model[column].mean(), inplace=True)
    elif method == "zero":
      for column in columns:
        self.df_model[column].fillna(0, inplace=True)

  def fix_column_str(self, column):
    self.df_model[column] = self.df_model[column].replace('', "0")

  def fix_columns_values(self,column, threshold):
    for col, thr in zip(column, threshold):
      self.df_model[col] = self.df_model[col].apply(lambda x: self.df_model[col].mean() if x < thr else x)

  def count_column_condition(self, column1, column2, condition):
    '''
      Count specific value in another column (column2) when we group by column2 
      given condition that aplies to column2

      Inputs:
        -column1: Column to group by
        -column2: Column in which we trying to count elements
        -condition: given condition to column2

      Returns:
        - List of columns and count of elements for them
    '''
    return self.df_model.groupby(column1)[column2].apply(lambda x: (x == condition).sum())

  def delete_columns(self, columns: List[str]):
    for i in columns:
      self.df_model = self.df_model.drop(i, axis=1)

  def dtypes(self):
    return self.df_model.dtypes

  def corr(self):
    plt.figure(figsize=(16,16))
    sns.heatmap(data=self.df_model.corr(), annot=True, fmt=".2f")

  def normal_distribution(self, columns):
    for i in columns:
      col = self.df_model[i]
      mean, std_dev = norm.fit(col)
      plt.figure(figsize=(10,10))
      plt.hist(col, bins=20, density=True, alpha=0.6, color='g')
      xmin, xmax = plt.xlim()
      x = np.linspace(xmin, xmax, 100)
      p = norm.pdf(x, mean, std_dev)
      plt.plot(x, p, 'k', linewidth=2)
      title = f"Normal distribution of {i}: mean = {mean:.2f}, std dev = {std_dev:.2f}"
      plt.title(title)
      plt.show()

  def normal_distribution_values(self, column, column_search):
    values = self.df_model[column].unique()
    for val in values:
      data_new = self.df_model[self.df_model[column] == val]
      col = data_new[column_search]
      mean, std_dev = norm.fit(col)
      plt.figure(figsize=(10,10))
      plt.hist(col, bins=20, density=True, alpha=0.6, color='g')
      xmin, xmax = plt.xlim()
      x = np.linspace(xmin, xmax, 100)
      p = norm.pdf(x, mean, std_dev)
      plt.plot(x, p, 'k', linewidth=2)
      title = f"Normal distribution of {column_search} by value {val}: mean = {mean:.2f}, std dev = {std_dev:.2f}"
      plt.title(title)
      plt.show()


  def normal_distribution_label(self,label, columns):
    ones_label = self.df_model[self.df_model[label] == 1]
    zero_label = self.df_model[self.df_model[label] == 0]
    for i in columns:
      plt.figure(figsize=(10,10))
      ones_label[i].plot(kind="kde", color="green", title=i)
      zero_label[i].plot(kind="kde", color="red")
      plt.xlim([0,max(ones_label[i])])

  def dummies(self, columns):
    df = pd.get_dummies(self.df_model, columns = columns)
    self.df_model = df


  def min_max_scaler(self, columns):
    clf = MinMaxScaler()
    df_min_max = self.df_model[columns]
    data_transformed = clf.fit_transform(df_min_max.to_numpy())
    data_transformed = pd.DataFrame(data_transformed, columns=columns)
    df_one_hot = pd.concat([self.df_model.drop(columns, axis=1), data_transformed], axis=1)
    self.df_model = df_one_hot

  def standard_scaler(self, columns):
    clf = StandardScaler()
    df_min_max = self.df_model[columns]
    data_transformed = clf.fit_transform(df_min_max.to_numpy())
    data_transformed = pd.DataFrame(data_transformed, columns=columns)
    df_one_hot = pd.concat([self.df_model.drop(columns, axis=1), data_transformed], axis=1)
    self.df_model = df_one_hot

  def best_features(self, n_features, steps, data_percentage, label_predict, type_problem, cache_size=5000):
    '''
      Uses RFE  to predict key features for model
      For type_problem = "regression" it uses SVR to extract key features
    '''
    if type_problem == "regression":
      df = self.df_model.sample(frac=data_percentage)
      X = df.drop(label_predict, axis=1)
      y = df[label_predict]
      estimator = SVR(kernel="linear", cache_size=cache_size)
      selector = RFE(estimator, n_features_to_select=n_features, step=steps)
      selector.fit(X,y)
      top_features = X.columns[selector.support_]
      return top_features