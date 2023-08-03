import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


diabetes_db = pd.read_csv("diabetes_train.csv")
diagnosis = diabetes_db["Outcome"]
db_train = diabetes_db.drop("Outcome", axis=1)


def create_decision_tree_model():
    clf = tree.DecisionTreeClassifier()
    clf.fit(db_train.values, diagnosis)

    return clf


def create_knn_model():
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(db_train.values, diagnosis)

    return neigh


def create_neural_network_model():
    clf = MLPClassifier(activation="logistic", random_state=1, max_iter=3000)
    clf.fit(db_train.values, diagnosis)

    return clf


def predict_diabetes_diagnosis(model, values):
    measurements = list(values.dict().values())

    result = model.predict([measurements])

    if result[0] == 1:
        return "Diagnosed with diabetes"
    else:
        return "Don't have diabetes"
