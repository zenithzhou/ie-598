#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:42:03 2018

@author: Zenith
"""

#load environment
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score

#load data
iris = datasets.load_iris()
X, y = iris.data, iris.target
# print(X,y)


# Part 1: test for range of random states
print('Part 1: \n')

randomstates = np.arange(1,11)
in_sample = []
out_sample = []

for i in randomstates:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=i)

    # SC = preprocessing.StandardScaler()
    # SC = SC.fit(X_train)

    dt = DecisionTreeClassifier(criterion = 'gini',random_state=42)
    dt.fit(X_train, y_train)

    y_pred_out = dt.predict(X_test)
    y_pred_in = dt.predict(X_train)

    out_sample.append(accuracy_score(y_test,y_pred_out))
    in_sample.append(accuracy_score(y_train,y_pred_in))

    print('Random State: ', i, '\n',
          'in_sample score:  %.4f'%in_sample[i-1],'\n',
          'out_sample score: %.4f'%out_sample[i-1],'\n')

# Plot
plt.scatter(randomstates, in_sample, c='red', label = 'In Sample')
plt.scatter(randomstates, out_sample, c='black', label = 'Out Sample')
plt.title('Random State vs in_sample and out_sample Accuracy')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower left')
# plt.show()

# Data Analysis
mean_in = np.mean(in_sample)
mean_out = np.mean(out_sample)
std_in = np.std(in_sample)
std_out = np.std(out_sample)
print('In sample Mean: %.3f, Out sample Mean: %.3f \n'
      'In sample STD: %.3f, Out sample STD: %.3f\n'%(mean_in, mean_out, std_in, std_out))



# Part 2: Different k-fold CV

print('Part 2: \n')

cv_scores = cross_val_score(dt,X_train,y_train, cv = 10)

print('10-fold CV scores:\n', cv_scores)

print("mean of cv score: %.4f"%np.mean(cv_scores))
print("variance of cv score: %.4f"%np.var(cv_scores))

y_pred = dt.predict(X_test)
print("out sample CV accuracy: %.4f"%accuracy_score(y_test, y_pred))


print("My name is Zenith Zhou")
print("My NetID is: zzhou64")
print("I hereby certify that I have read the University policy on "
      "Academic Integrity and that I am not in violation.")
