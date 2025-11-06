import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters = 3, tol = 1e-4, random_state = 335, max_iter = 100):
        self.n_clusters = n_clusters
        self.tol = tol
        self.random_state = random_state
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    def fit(self, X):
        idx = np.random.choice(X.shape[0], self.n_clusters, replace = False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            distances = cdist(X, self.centroids, 'euclidean')
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[self.labels == k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)