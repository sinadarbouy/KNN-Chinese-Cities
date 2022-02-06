from sklearn import datasets
from sklearn.model_selection import train_test_split
from Classification import Classifier
# from sklearn.neighbors import KNeighborsClassifier

# iris = datasets.load_iris()


# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, random_state=4, test_size=0.3)

# clf = KNeighborsClassifier(n_neighbors=25, metric="euclidean")
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)


# print(clf.score(X_test, y_test))

# clf = Classifier(n_neighbors=3, metric="euclidean")
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
# for i in prediction:
#     print(i, end=' ')

# print(clf.score(X_test, y_test))


# from Classification import Classifier
# import numpy as np
import pandas as pd

Beijing_labeled = pd.read_csv(
    "C:\\Sina\\uni\AI System\\Assignment 3\\KNN-Chinese-Cities\\Cities\\Beijing_labeled.csv")
Shenyang_labeled = pd.read_csv(
    "C:\\Sina\\uni\AI System\\Assignment 3\\KNN-Chinese-Cities\\Cities\\Shenyang_labeled.csv")
dataframes = [Beijing_labeled, Shenyang_labeled]
Beijing_Shenyang_labels = pd.concat(dataframes)
Shanghai_labeled = pd.read_csv(
    "C:\\Sina\\uni\AI System\\Assignment 3\\KNN-Chinese-Cities\\Cities\\Shanghai_labeled.csv")
Guangzhou_labeled = pd.read_csv(
    "C:\\Sina\\uni\AI System\\Assignment 3\\KNN-Chinese-Cities\\Cities\\Guangzhou_labeled.csv")


feature_columns = ['season', 'DEWP', 'HUMI', 'PRES', 'TEMP',
                   'Iws', 'precipitation', 'cbwd_NE']
X_train = Beijing_Shenyang_labels[feature_columns].values
y_train = Beijing_Shenyang_labels['PM_HIGH'].values
x_test_Guangzhou = Guangzhou_labeled[feature_columns].values
y_test_Guangzhou = Guangzhou_labeled['PM_HIGH'].values

# clf = KNeighborsClassifier(n_neighbors=25, metric="euclidean")
clf = Classifier(n_neighbors=25, metric="euclidean")

clf.fit(X_train, y_train)
prediction = clf.predict(x_test_Guangzhou)

print(clf.score(x_test_Guangzhou, y_test_Guangzhou))

# max = 0
# for i in range(9, 0, -1):
#     selected_features = feature_columns[:i]
#     X_train = Beijing_Shenyang_labels[selected_features].values
#     y_train = Beijing_Shenyang_labels['PM_HIGH'].values
#     x_test_Guangzhou = Guangzhou_labeled[feature_columns].values
#     y_test_Guangzhou = Guangzhou_labeled['PM_HIGH'].values
#     c = Classifier(n_neighbors=15, metric="euclidean")
#     c.fit(X_train, y_train)
#     prediction = c.predict(x_test_Guangzhou)
#     score = c.score(x_test_Guangzhou, y_test_Guangzhou)
#     if score > max:
#         max = score
#         out = i
#         print("n_features: " + str(out))
#         print("score: " + str(max) + "\n")
