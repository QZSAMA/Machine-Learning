from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


iris = datasets.load_iris()
X = iris.data[:, [0, 1, 2, 3]]
y = iris.target
# print(X)
# print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred=svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(X_train)
fig, ax = plt.subplots()

plot_decision_regions(X_train_std, y_train, clf=svm,
                      filler_feature_values={3: 0.25,2:0.25},
                      filler_feature_ranges={3: 6,2:6},
                      res=0.02, legend=2, ax=ax)
ax.set_xlabel('Sepal Length[cm]')
ax.set_ylabel('Sepal Width[cm]')
ax.set_title('Petal Length[cm] =2 +- 2  and Petal width =  3.5 +- 3.5')

# Adding axes annotations
fig.suptitle('SVM linear kernal on Sepal Length and Width')
plt.show()



svm = SVC(kernel='rbf', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred=svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
fig, ax = plt.subplots()

plot_decision_regions(X_train_std, y_train, clf=svm,
                      filler_feature_values={3: 0.25,2:0.25},
                      filler_feature_ranges={3: 6,2:6},
                      res=0.02, legend=2, ax=ax)
ax.set_xlabel('Sepal Length[cm]')
ax.set_ylabel('Sepal Width[cm]')
ax.set_title('Petal Length[cm] =0.25 +- 6  and Petal width =  0.25 +- 6')

# Adding axes annotations
fig.suptitle('SVM rbf kernal on Sepal Length and Width')
plt.show()


svm = SVC(kernel='poly', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred=svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='sigmoid', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred=svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
