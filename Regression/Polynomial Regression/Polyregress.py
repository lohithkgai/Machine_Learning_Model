# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:14:49 2019

@author: Tonyopt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("Position_Salaries.csv")

X=dataset.drop(columns=["Position","Salary"])
y=dataset.Salary

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Linear regression result visualization
lm_pred=lm.predict(X)
plt.scatter(X,y,color="red")
plt.plot(X,lm_pred,color="blue")

#Poly regression result visualization
lin_reg_pred=lin_reg_2.predict(X_poly)
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg_pred,color="blue")
plt.show()

#
lm_prediction=lm.predict([[6.5]])
poly_prediction=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))