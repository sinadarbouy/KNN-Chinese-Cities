from math import sqrt, pow


class Calculation():
    @staticmethod
    def calculatedistance(metric, x, y):
        _metric = metric.lower()
        if _metric == "hamming":
            return hamming_distance(x, y)
        elif _metric == "euclidean":
            return euclidean_distance(x, y)
        else:
            return euclidean_distance(x, y)


# Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    distance = 0.0
    len_vec_1 = len(vec1)
    for i in range(len_vec_1):
        distance += pow((vec1[i] - vec2[i]), 2)
    euclidean_value = sqrt(distance)
    return euclidean_value

# hamming distance between two string for binary input


def hamming_distance(string1, string2):
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter
