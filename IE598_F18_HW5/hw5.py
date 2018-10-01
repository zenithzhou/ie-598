import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#load data
df_wine = pd.read_csv('/Users/Zenith/Desktop/OneDrive/School/FA18/IE598/HWs/hw5/wine.csv',
                      header=0)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())
print(df_wine.describe())


# 1. Exploratory Data Analysis
print("\n #1. EDA")

# Scatterplot matrix of first 5 features
sns.pairplot(df_wine[df_wine.columns[:5]], size=2)
plt.tight_layout()
plt.show()

# Create heatmap
corMat=DataFrame(df_wine.corr())
plt.pcolor(corMat)
plt.show()

sns.heatmap(corMat,cbar=True, annot=True, square=True,
            fmt='.2f',annot_kws={'size':6.5},yticklabels=df_wine.columns,xticklabels=df_wine.columns)
plt.show()


# assign X and y
X = df_wine.iloc[:, :-1].values
y = df_wine.iloc[:,-1].values

# print(X,y)
# standardize the features
standscaler = StandardScaler()
standscaler.fit(X)
X_processed = standscaler.transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Decision Regions Plotter
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
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
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

print("\nEDA plots exported")


# 2. Logistic Regression Classifier & SVM
print("\n#2. Logistic Regression & SVM \n")
# Logistic regression
lr=LogisticRegression(C=10.0, random_state=42)
lr.fit(X_train,y_train)
y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)
print( "Logistic regression train score:",metrics.accuracy_score(y_train, y_train_pred))
print( "Logistic regression test score:",metrics.accuracy_score(y_test, y_test_pred))


# SVM Classifier (Baseline)
svm=SVC(kernel='linear',C=1.0)
svm.fit(X_train,y_train)
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)
print( "SVM train score:",metrics.accuracy_score(y_train, y_train_pred))
print( "SVM test score:",metrics.accuracy_score(y_test, y_test_pred))


# 3. PCA transform
print("\n#3 PCA transformation\n")

# PCA
pca=PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Post PCA Logistic regression
lr.fit(X_train_pca, y_train)
y_train_pred=lr.predict(X_train_pca)
y_test_pred=lr.predict(X_test_pca)
print( "(PCA)Logistic regression train score:",metrics.accuracy_score(y_train, y_train_pred))
print( "(PCA)Logistic regression test score:",metrics.accuracy_score(y_test, y_test_pred))

plt.figure()
plot_decision_regions(X_test_pca, y_test, lr, resolution=0.02)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Post PCA Logistic Regression')
plt.legend(loc='lower left')

plt.show()


# Post PCA SVM
svm.fit(X_train_pca, y_train)
y_train_pred=svm.predict(X_train_pca)
y_test_pred=svm.predict(X_test_pca)
print("(PCA)SVM train score:",metrics.accuracy_score(y_train, y_train_pred))
print("(PCA)SVM test score:",metrics.accuracy_score(y_test, y_test_pred))

plt.figure()
plot_decision_regions(X_test_pca, y_test, svm, resolution=0.02)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Post PCA SVM')
plt.legend(loc='lower left')

plt.show()


# 4. LDA transform
print("\n#4 LDA transformation\n")


## LDA
lda=LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Post LDA Logistic Regression
lr = lr.fit(X_train_lda, y_train)
y_train_pred=lr.predict(X_train_lda)
y_test_pred=lr.predict(X_test_lda)

print("(LDA)Logistic regression train score:",metrics.accuracy_score(y_train, y_train_pred))
print("(LDA)Logistic regression test score:",metrics.accuracy_score(y_test, y_test_pred))

plt.figure()
plot_decision_regions(X_test_lda, y_test, lr, resolution=0.02)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Post LDA Logistic Regression')
plt.legend(loc='lower left')
plt.show()


# Post LDA SVM
svm.fit(X_train_lda, y_train)
y_train_pred=svm.predict(X_train_lda)
y_test_pred=svm.predict(X_test_lda)
print("(LDA)SVM train score:",metrics.accuracy_score(y_train, y_train_pred))
print("(LDA)SVM test score:",metrics.accuracy_score(y_test, y_test_pred))

plt.figure()
plot_decision_regions(X_test_lda, y_test, svm, resolution=0.02)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Post LDA SVM')
plt.legend(loc='lower left')

plt.show()


print("\n#5 kPCA transformation\n")

gamma_range = [0.01, 0.1, 0.5, 1, 5]

test_scores = {}

for g in gamma_range:
    kpca = KernelPCA(n_components=2, kernel='rbf',gamma=g)
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)

    #post kPCA Logistic Regression
    lr.fit(X_train_kpca, y_train)
    y_train_pred = lr.predict(X_train_kpca)
    y_test_pred = lr.predict(X_test_kpca)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)

    test_scores[g] = test_score

    print("(KPCA)Logistic regression train score at gamma = ",g,": ",train_score)
    print("(KPCA)Logistic regression test score at gamma = ",g,": ",test_score)
    plt.figure()
    plot_decision_regions(X_test_kpca, y_test, lr, resolution=0.02)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('post kPCA Logistic Regression at gamma = %f'%g)
    plt.legend(loc='lower left')
    plt.show()

    #post kPCA SVM
    svm = SVC(kernel = 'rbf', C = 1.0)
    svm.fit(X_train_kpca, y_train)
    y_train_pred = svm.predict(X_train_kpca)
    y_test_pred = svm.predict(X_test_kpca)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)

    print( "(KPCA)SVM train score at gamma = ",g,": ",train_score)
    print( "(KPCA)SVM test score at gamma = ",g,": ",test_score)

    print("\n")
    plt.figure()
    plot_decision_regions(X_test_kpca, y_test, svm, resolution=0.02)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Post PCA SVM at gamma = %f'%g)
    plt.legend(loc='lower left')
    plt.show()

print("Best gamma is ",max(test_scores, key = test_scores.get), "\n")

print("My name is Zenith Zhou")
print("My NetID is: zzhou64")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
