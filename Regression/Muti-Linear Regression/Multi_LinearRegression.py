# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:46:21 2019

@author: Tonyopt
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")
dataset.info()

dummy=pd.get_dummies(dataset.State,drop_first=True)
newdata=pd.concat([dataset,dummy],axis=1)
feat=newdata.drop(columns="State")

X=feat.drop(columns="Profit").values
y=feat.Profit.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)

predictions=lm.predict(X_test)

from sklearn import metrics
MSE=metrics.mean_squared_error(y_test,predictions)
RMSE=np.sqrt(MSE)
EVS=metrics.explained_variance_score(y_test, predictions)
print(f"MSE:{MSE}")
print(f"RMSE:{RMSE}")
print(f"Explained Variance Score:{EVS}")

import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
