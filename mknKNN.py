from scipy.stats import mode
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, n_neighbours, metric):
        self.n_neighbours = n_neighbours
        self.metric = metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        distances = cdist(X_test, self.X_train, self.metric)
        y_test = []
        for i in range(0, distances.shape[0]):
            labels = np.argpartition(distances[i], self.n_neighbours)[:self.n_neighbours]
            y_test.append(mode(self.y_train[labels])[0])
        return y_test