#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 22:10:08 2018

@author: Zenith
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,
                                                    random_state=1,
                                                    stratify=y)

esti_range = [1,2,3,10,20,50,200]

for i in esti_range:
    forest = RandomForestClassifier(n_estimators=i,
                                    random_state=1)
    
    forest.fit(X_train,y_train)
    
    in_sam_score = forest.score(X_train,y_train)
    out_sam_score = forest.score(X_test,y_test)
    
    print("In-sample-score = ", in_sam_score, 
          ", for estimator ",i)
    
    print("Out-sample-score= ", out_sam_score, 
          ", for estimator= ",i,'\n')
    
    
    
#    print(score)
forest = RandomForestClassifier(n_estimators=1000,
                                    random_state=1)
forest.fit(X_train,y_train)

print("CV_score:")
CV_score = cross_val_score(forest, X_train, y_train, cv=10)
print(CV_score)


feat_labels = df_wine.columns[1:]

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='blue')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

    
print("My name is Zenith Zhou")
print("My NetID is: zzhou64")
print("I hereby certify that I have read the University policy on "
      "Academic Integrity and that I am not in violation.")