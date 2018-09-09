from sklearn import datasets
import numpy as np
import pandas as pd
import operator

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# import iris data
iris = datasets.load_iris()

origin = pd.read_excel("C://Users/yz_ze/OneDrive/School/FA18/IE598/HWs/hw2/codes/data/Cat_or_Dog.xlsx", header=None)
datas = pd.DataFrame(origin)


# define initial X and y
X = iris.data
y = iris.target
# y = datas[4][1:].values
# X = datas[:][1:].values


# check class labels
print('Data set includes following class labels: ', np.unique(y))

# split training and testing sets by 30%/70%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2, stratify=y)


def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
    # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')



# test highest accuracy score K
training_scores = {}
test_scores = {}

#dict that keeps all the scores
for k in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    name = str(k)
    training_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)

    training_scores[name] = training_score
    test_scores[name] = test_score

sorted_train_scores = sorted(training_scores.items(), key = operator.itemgetter(1))
sorted_test_scores = sorted(test_scores.items(), key = operator.itemgetter(1))

highest_K = sorted_train_scores[-2]
print("Highest score comes with k = " + highest_K[0] + " with score " + str(training_scores[highest_K[0]]))

print(sorted_train_scores)
print(sorted_test_scores)

# fit knn with best K
knn = KNeighborsClassifier(n_neighbors=int(highest_K[0]))
knn.fit(X_train, y_train)


#Decision tree
dt = DecisionTreeClassifier(
    max_depth=6,
    random_state=1
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
score = accuracy_score(y_test,y_pred)

print("accuracy score for dt is: "+str(score))


# plotting of data
train_neighbors = []
test_neighbors = []
train_accuracy = []
test_accuracy = []

for value in training_scores.values():
    train_accuracy.append(value)

for value in training_scores.keys():
    train_neighbors.append(int(value))

for value in test_scores.values():
    test_accuracy.append(value)

for value in test_scores.keys():
    test_neighbors.append(int(value))


plt.title('k-NN: Varing Number of Neighbors')
plt.plot(train_neighbors,train_accuracy, label = 'Training Accuracy')
plt.plot(test_neighbors,test_accuracy, label = 'Testingt Accuracy')

plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')

plt.show()

print("My name is Ziheng Zhou")
print("My NetID is: zzhou64")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")