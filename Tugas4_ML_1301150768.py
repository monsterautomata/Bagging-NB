# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 07:48:49 2019

@author: Rozan
"""

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#import data training
data = pd.read_csv("TrainsetTugas4ML.csv")
x = data.iloc[:,0:2]
y = data.iloc[:,2:3]

#import data testing
datatest = pd.read_csv("TestsetTugas4ML.csv")
x_test = data.iloc[:,0:2]
y_test = data.iloc[:,2:3]

#kfold
seed = 10
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#Naive Bayes
cart = GaussianNB()
cart.fit(x,y)
y_testnb = cart.predict(x_test)
accuracynb = accuracy_score(y, y_testnb) * 100
result1 = model_selection.cross_val_score(cart,x,y,cv=kfold)
kfold1=result1.mean() * 100


#bagging
clf = BaggingClassifier(base_estimator=cart, n_estimators=10, random_state=seed)
clf.fit(x,y)
y_test = clf.predict(x_test)
accuracybag=accuracy_score(y, y_test) * 100
result2 = model_selection.cross_val_score(clf,x,y,cv=kfold)
kfold2=result2.mean() * 100


datatest['Class']=y_test[:75]
datatest.to_csv('TebakanTugas4ML.csv', index=False, header=None)

