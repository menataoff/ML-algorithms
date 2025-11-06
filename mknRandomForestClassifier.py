import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class RandomForestClassifierOnSKL():
    def __init__(self, n_estimators = 100, max_features = 'sqrt', max_depth = 50, bootstrap = True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(len(self.Xtrain), size=len(self.Xtrain), replace=True)
            else:
                indices = np.arange(len(self.Xtrain))
            tree = DecisionTreeClassifier(
            max_features = self.max_features,
            max_depth = self.max_depth
            )
            X1 = self.Xtrain[indices]
            y1 = self.ytrain[indices]
            tree.fit(X1, y1)
            self.trees.append(tree)
    def predict(self, Xtest):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(Xtest))
        preds = np.array(predictions)
        return mode(preds)[0]
