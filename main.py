from fastapi import FastAPI

from models import (
    create_decision_tree_model,
    create_knn_model,
    create_neural_network_model,
    predict_diabetes_diagnosis
)
from scores import (
    score_for_decision_tree,
    score_for_knn,
    score_for_neural_network
)
from serializers import DiagnosticMeasurements

app = FastAPI()

decision_tree_model = create_decision_tree_model()
knn_model = create_knn_model()
neural_network_model = create_neural_network_model()


@app.get("/")
async def root():
    return {"message": "API for diabetes diagnosis prediction"}


@app.get("/scores")
async def get_scores():
    decision_tree_score = score_for_decision_tree()
    knn_score = score_for_knn()
    neural_network_score = score_for_neural_network()

    return {
        "decision_tree_score": decision_tree_score,
        "knn_score": knn_score,
        "neural_network_score": neural_network_score
    }


@app.post("/decision-tree-diagnosis")
async def get_decision_tree_diagnosis(measurements: DiagnosticMeasurements):
    diagnosis = predict_diabetes_diagnosis(decision_tree_model, measurements)

    return {"diagnosis": diagnosis}


@app.post("/knn-diagnosis")
async def get_knn_diagnosis(measurements: DiagnosticMeasurements):
    diagnosis = predict_diabetes_diagnosis(knn_model, measurements)

    return {"diagnosis": diagnosis}


@app.post("/neural-network-diagnosis")
async def get_neural_network_diagnosis(measurements: DiagnosticMeasurements):
    diagnosis = predict_diabetes_diagnosis(neural_network_model, measurements)

    return {"diagnosis": diagnosis}
