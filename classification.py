from math import sqrt, pow
import scipy.spatial


class Classifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        pass

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X_test):
        return _predict(self, X_test)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return _score(predictions, y_test)


# Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    # return scipy.spatial.distance.euclidean(X_train[j], X_test[i])
    distance = 0.0
    len_vec_1 = len(vec1)
    for i in range(len_vec_1):
        distance += pow((vec1[i] - vec2[i]), 2)
    euclidean_value = sqrt(distance)
    return euclidean_value


# find neares neighbors
def find_neighbors(train, test, n_neighbors):
    distances = [(index_train_item, euclidean_distance(test, train_item))
                 for index_train_item, train_item in enumerate(train)]
    distances.sort(key=lambda x: x[1])  # sort distances
    neighbors = [i[0] for i in distances[:n_neighbors]]
    return neighbors


def _predict(self, X_test):
    predictions = []
    for test_data in X_test:
        neighbors = find_neighbors(self.X_train, test_data, self.n_neighbors)
        y_neighbors = [self.y_train[i]
                       for i in neighbors]  # get y_train value of neighbors
        # get a highest probability value
        prediction = max(y_neighbors, key=y_neighbors.count)
        predictions.append(prediction)
    return predictions


def _score(predictions, y_test):
    print('predictions ____________________')
    for i in predictions:
        print(i, end=' ')
    count = (predictions == y_test).sum()
    score = count / len(y_test)
    return score * 100.00
