import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

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

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

ab = df.iloc[50:150, 4].values
ab = np.where(ab == 'Iris-versicolor', -1, 1)

# # #Case 0 sepal length and petal length---------------------------------------------------------
# Case0= df.iloc[50:150, [0, 2]].values

# # plot data
# plt.scatter(Case0[:50, 0], Case0[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case0[50:100, 0], Case0[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()

# X_std = np.copy(Case0)
# X_std[:, 0] = (Case0[:, 0] - Case0[:, 0].mean()) / Case0[:, 0].std()
# X_std[:, 1] = (Case0[:, 1] - Case0[:, 1].mean()) / Case0[:, 1].std()

# ada0 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada0.fit(X_std, ab)

# plot_decision_regions(X_std, ab, classifier=ada0)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada0.cost_) + 1), ada0.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()




# #Case 1 sepal length and sepal width---------------------------------------------------------
# Case1= df.iloc[50:150, [0, 1]].values
# # print(Case1)
# # plot data
# plt.scatter(Case1[:50, 0], Case1[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case1[50:100, 0], Case1[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('sepal length [cm]')
# plt.ylabel('sepal width [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()
# X_std1 = np.copy(Case1)
# X_std1[:, 0] = (Case1[:, 0] - Case1[:, 0].mean()) / Case1[:, 0].std()
# X_std1[:, 1] = (Case1[:, 1] - Case1[:, 1].mean()) / Case1[:, 1].std()
# ada1 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada1.fit(X_std1, ab)

# plot_decision_regions(X_std1, ab, classifier=ada1)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('sepal width [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()

# #Case 2 sepal length and petal width-------------------------------------------------------------------
# Case2= df.iloc[50:150, [0, 3]].values

# # plot data
# plt.scatter(Case2[:50, 0], Case2[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case2[50:100, 0], Case2[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()

# X_std2 = np.copy(Case2)
# X_std2[:, 0] = (Case2[:, 0] - Case2[:, 0].mean()) / Case2[:, 0].std()
# X_std2[:, 1] = (Case2[:, 1] - Case2[:, 1].mean()) / Case2[:, 1].std()
# ada2 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada2.fit(X_std2, ab)

# plot_decision_regions(X_std2, ab, classifier=ada2)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()

# #Case 3 sepal width and petal length --------------------------------------------------------------------------------
# Case3= df.iloc[50:150, [1, 2]].values

# # plot data
# plt.scatter(Case3[:50, 0], Case3[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case3[50:100, 0], Case3[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('sepal width [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)

# X_std3 = np.copy(Case3)
# X_std3[:, 0] = (Case3[:, 0] - Case3[:, 0].mean()) / Case3[:, 0].std()
# X_std3[:, 1] = (Case3[:, 1] - Case3[:, 1].mean()) / Case3[:, 1].std()
# plt.show()

# ada3 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada3.fit(X_std3, ab)

# plot_decision_regions(X_std3, ab, classifier=ada3)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal width [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()

# #Case 4 sepal width and petal width --------------------------------------------------------------------------------
# Case4= df.iloc[50:150, [1, 3]].values

# # plot data
# plt.scatter(Case4[:50, 0], Case4[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case4[50:100, 0], Case4[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('sepal width [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()
# X_std4 = np.copy(Case4)
# X_std4[:, 0] = (Case4[:, 0] - Case4[:, 0].mean()) / Case4[:, 0].std()
# X_std4[:, 1] = (Case4[:, 1] - Case4[:, 1].mean()) / Case4[:, 1].std()

# ada4 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada4.fit(X_std4, ab)

# plot_decision_regions(X_std4, ab, classifier=ada4)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal width [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada4.cost_) + 1), ada4.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()

# #Case 5 petal length and petal width --------------------------------------------------------------------------------
# Case5= df.iloc[50:150, [2, 3]].values

# # plot data
# plt.scatter(Case5[:50, 0], Case5[:50, 1],
#             color='red', marker='o', label='versicolor')
# plt.scatter(Case5[50:100, 0], Case5[50:100, 1],
#             color='blue', marker='x', label='virginica')



# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')

# # plt.savefig('images/02_06.png', dpi=300)
# plt.show()
# X_std5 = np.copy(Case5)
# X_std5[:, 0] = (Case5[:, 0] - Case5[:, 0].mean()) / Case5[:, 0].std()
# X_std5[:, 1] = (Case5[:, 1] - Case5[:, 1].mean()) / Case5[:, 1].std()
# ada5 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada5.fit(X_std5, ab)

# plot_decision_regions(X_std5, ab, classifier=ada5)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada5.cost_) + 1), ada5.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()

# Case 6 sepal length ,sepal width and petal length---------------------------------------------------------
Case6= df.iloc[50:150, [0, 1, 2]].values
# print(Case6)
# plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Case6[:50, 0], Case6[:50, 1],Case6[:50, 2],
            color='red', marker='o', label='versicolor')
ax.scatter(Case6[50:100, 0], Case6[50:100, 1],Case6[50:100, 1],
            color='blue', marker='x', label='virginica')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# plt.savefig('images/02_06.png', dpi=300)
plt.show()

X_std6 = np.copy(Case6)
X_std6[:, 0] = (Case6[:, 0] - Case6[:, 0].mean()) / Case6[:, 0].std()
X_std6[:, 1] = (Case6[:, 1] - Case6[:, 1].mean()) / Case6[:, 1].std()
X_std6[:, 2] = (Case6[:, 2] - Case6[:, 2].mean()) / Case6[:, 2].std()

ada6 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
print(X_std6)
# ada6.fit(X_std6, ab)

# ax.plot(range(1, len(ada6.cost_) + 1), ada6.cost_, marker='o')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()
