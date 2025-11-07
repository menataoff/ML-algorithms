import numpy as np


class mknGradientDescent:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-5):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.loss_history = []

    def compute_gradient(self, X, y, w):
        grad = (1/y.shape[0]) * X.T @ (X @ w - y)
        self.loss_history.append(np.linalg.norm(grad))
        return grad

    def fit(self, X, y, w):
        for i in range(self.max_iters):
            grad = self.compute_gradient(X, y, w)
            w_new = w - self.learning_rate * grad

            # Проверка сходимости
            if np.abs(w_new - w) < self.tol:
                break
            w = w_new
        return w