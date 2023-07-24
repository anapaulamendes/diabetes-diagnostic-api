from fastapi import FastAPI

from neural_network_model import (
    create_neural_network_model,
    predict_diabetes_diagnosis
)
from serializers import DiagnosticMeasurements

app = FastAPI()

neural_network_model = create_neural_network_model()


@app.get("/")
async def root():
    return {"message": "API for diabetes diagnosis prediction"}


@app.post("/neural-network-diagnosis")
async def get_neural_network_diagnosis(measurements: DiagnosticMeasurements):
    diagnosis = predict_diabetes_diagnosis(neural_network_model, measurements)

    return {"diagnosis": diagnosis}
