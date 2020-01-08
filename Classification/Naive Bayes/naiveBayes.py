# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:02:23 2019

@author: Tonyopt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("Social_Network_Ads.csv")
dataset.info()

dummy=pd.get_dummies(dataset.Gender,drop_first=True)
newData=pd.concat([dataset,dummy],axis=1)
feature=newData.drop(columns="Gender")

X=feature.drop(columns=["Purchased","User ID"]).values
y=feature.Purchased.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))