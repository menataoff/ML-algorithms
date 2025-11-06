from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoostingRegressionMKN:
    def __init__(self, learning_rate=0.1, M = 200, max_depth = 6):
        self.learning_rate = learning_rate
        self.M = M
        self.max_depth = max_depth
        self.trees = []
    def fit(self, X, y):
        #В данном алгоритме функция потерь - MSE = 1/2 * (y - F(x))**2
        #Антиградиент = y - F(x)
        if (X.ndim == 1):
            self.X = X.reshape(-1, 1)
        else:
            self.X = X
        self.y = y
        self.F = np.ones(self.X.shape[0]) * np.mean(self.y)
        for i in range(self.M):
            residuals = self.y - self.F
            tree = DecisionTreeRegressor(max_depth = self.max_depth)
            tree.fit(self.X, residuals)
            self.trees.append(tree)
            self.F += self.learning_rate * residuals
    def predict(self, X):
        if (X.ndim == 1):
            X = X.reshape(-1, 1)
        pred = np.ones(X.shape[0]) * np.mean(self.y)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred