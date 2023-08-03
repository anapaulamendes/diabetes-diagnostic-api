import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


diabetes_db = pd.read_csv("diabetes_complete.csv")
X = diabetes_db.drop("Outcome", axis=1)
y = diagnostic = diabetes_db["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1
)


def score_for_decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score


def score_for_knn():
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)

    return score


def score_for_neural_network():
    clf = MLPClassifier(activation="logistic", random_state=1, max_iter=3000)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
