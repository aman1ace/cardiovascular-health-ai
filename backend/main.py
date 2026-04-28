from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_health

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
risk_model = joblib.load("risk_model.pkl")
score_model = joblib.load("score_model.pkl")

# Input schema


class InputData(BaseModel):
    age: int
    before_hr: int
    after_hr: int
    max_hr: int
    sleep: float


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: InputData):

    return predict_health(
        data.age,
        data.before_hr,
        data.after_hr,
        data.max_hr,
        data.sleep,
        risk_model,
        score_model
    )
