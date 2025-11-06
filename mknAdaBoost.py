from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoostMKN:
    def __init__(self, n_estimators = 50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.classifiers = []
        self.weights_history = []
    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.N = self.Xtrain.shape[0]
        self.weights = np.ones(self.N) / self.N
        for i in range(self.n_estimators):
            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.fit(Xtrain, ytrain, sample_weight=self.weights)
            predictions = classifier.predict(Xtrain)
            incorrect = (predictions != ytrain)
            error = np.sum(self.weights[incorrect])  # ε_t
            if error >= 0.5:  # Классификатор хуже случайного
                break  # Прекращаем обучение

            if error <= 1e-10:  # Идеальный классификатор
                alpha = 100.0
            else:
                alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # +ϵ для стабильности
            self.weights *= np.exp(alpha * incorrect)  # Увеличиваем веса ошибочных объектов
            self.weights /= np.sum(self.weights)
            self.weights_history.append(self.weights)
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.classifiers):
            final_predictions += alpha * classifier.predict(X)
        return np.sign(final_predictions)