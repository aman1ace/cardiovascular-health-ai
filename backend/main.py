from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_health
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI(title="Cardiovascular AI API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": exc.errors()}
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

risk_model = joblib.load("risk_model.pkl")
score_model = joblib.load("score_model.pkl")


class InputData(BaseModel):
    age: int
    before_hr: int
    after_hr: int
    max_hr: int
    sleep: float


@app.get("/")
def home():
    return {
        "message": "Cardiovascular AI API is running",
        "endpoints": ["/predict", "/watch-predict", "/smartwatch-predict"]
    }


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


# Health Connect / Android bridge endpoint
@app.post("/watch-predict")
def watch_predict(data: InputData):
    return predict_health(
        data.age,
        data.before_hr,
        data.after_hr,
        data.max_hr,
        data.sleep,
        risk_model,
        score_model
    )


# Demo fallback endpoint
@app.get("/smartwatch-predict")
def smartwatch_predict():
    age = 25
    before_hr = 88
    after_hr = 76
    max_hr = 165
    sleep = 7.2

    return predict_health(
        age,
        before_hr,
        after_hr,
        max_hr,
        sleep,
        risk_model,
        score_model
    )
