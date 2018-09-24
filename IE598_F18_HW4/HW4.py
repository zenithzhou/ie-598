#import necessary repos

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression


# import housing dataset
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,sep='\s+')


df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM', 'AGE','DIS',
              'RAD','TAX','PTRATIO','B','LSTAT','MEDV']

## EDA
print(df.describe())
print(df.info())
print(df.describe())


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']


# Basic Scatter Plot
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

# Seaborn's Heatmap
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()


## Data set train-test split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,
                                                    random_state=42)


##Scikit Learn Linear Regression
slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', 
            edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', 
            edgecolor='white', label='Test data')

plt.title('Linear Model')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),
                                      r2_score(y_test, y_test_pred)))

features = list(df.columns)[:-1]
features.append('intercept')
lr_coefs = np.zeros(X.shape[1]+1)
lr_coefs[:-1] = slr.coef_
lr_coefs[-1] = slr.intercept_
print(pd.DataFrame(lr_coefs, index = features, columns = ['lr_coefs']))


print("\n")


## Ridge Model
print("Ridge: \n")

alpha = [0.1, 0.5, 1.0, 2.0]


for i in range(len(alpha)):
    ridge = Ridge(alpha=alpha[i])
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    plt.scatter(y_train_pred, y_train_pred - y_train,
           c='blue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
           c='yellow', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('ridge Residuals')
    plt.legend(loc='upper left')
    plt.title("Ridge Performance at alpha = " + str(alpha[i]))
    plt.hlines(y=0, xmin=0, xmax=50, color='black', lw=2)
    plt.xlim([-10,50])
    plt.show()
    print('At alpha = '+ str(alpha[i])+', MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                                         mean_squared_error(y_test, y_test_pred)))
    print(
        '                R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

print("\n")


# Lasso Regression

print("Lasso: \n")

alpha = [0.1, 0.5, 1.0, 2.0]

for i in range(len(alpha)):
    lasso = Lasso(alpha=alpha[i])
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='blue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='yellow', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Lasso Residuals')
    plt.legend(loc='upper left')
    plt.title("Lasso Performance at alpha = " + str(alpha[i]))
    plt.hlines(y=0, xmin=0, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    print('At alpha = ' + str(alpha[i]) + ', MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                                                             mean_squared_error(y_test, y_test_pred)))
    print(
        '                R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

print("\n")


# ElasticNet Regression

print("Elastic Net: \n")

l1_ratio = [0.1, 0.3, 0.5, 0.8]

for i in range(len(l1_ratio)):
    elanet = ElasticNet(alpha = 0.1, l1_ratio=l1_ratio[i])
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)

    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='blue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='yellow', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Elastic Net Residuals')
    plt.legend(loc='upper left')
    plt.title("Elastic Net Performance at L1 = " + str(l1_ratio[i]))
    plt.hlines(y=0, xmin=0, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    print('At alpha = 0.1, L1 ratio of ' + str(l1_ratio[i]) + ', MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                                                             mean_squared_error(y_test, y_test_pred)))
    print(
        '                                 R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

print("\n")


print("-------------------the end-------------------")


print("My name is Zenith Zhou")
print("My NetID is: zzhou64")
print("I hereby certify that I have read the University policy "
      "on Academic Integrity and that I am not in violation.")

