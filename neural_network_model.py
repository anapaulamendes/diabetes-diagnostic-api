import pandas as pd
from sklearn.neural_network import MLPClassifier


def create_neural_network_model():
    diabetes_db = pd.read_csv("diabetes_train.csv")
    diagnosis = diabetes_db["Outcome"]
    db_train = diabetes_db.drop("Outcome", axis=1)

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
