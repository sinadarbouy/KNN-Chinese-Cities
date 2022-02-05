from Calculation import Calculation
import scipy.spatial


class Classifier():
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric
        pass

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X_test):
        return _predict(self, X_test)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return _score(predictions, y_test)


# find neares neighbors
def find_neighbors(self, train, test):
    distances = [(index_train_item, Calculation.calculatedistance(self.metric, test, train_item))
                 for index_train_item, train_item in enumerate(train)]
    distances.sort(key=lambda x: x[1])  # sort distances
    neighbors = [i[0] for i in distances[:self.n_neighbors]]
    return neighbors


def _predict(self, X_test):
    predictions = []
    for test_data in X_test:
        neighbors = find_neighbors(self, self.X_train, test_data)
        y_neighbors = [self.y_train[i]
                       for i in neighbors]  # get y_train value of neighbors
        # get a highest probability value
        prediction = max(y_neighbors, key=y_neighbors.count)
        predictions.append(prediction)
    return predictions


def _score(predictions, y_test):
    count_Similar = (predictions == y_test).sum()
    score = count_Similar / len(y_test)
    return score * 100.00
