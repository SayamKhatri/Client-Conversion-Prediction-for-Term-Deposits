from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from google.cloud import aiplatform, storage
import pandas as pd
import pickle
from typing import List
import os
from dotenv import load_dotenv

app = FastAPI()

PROJECT_ID = os.environ['PROJECT_ID']
ENDPOINT_ID = os.environ['ENDPOINT_ID']
REGION = os.environ['REGION']
BUCKET_NAME = os.environ['BUCKET_NAME']
LABEL_ENCODER_PATH = os.environ['LABEL_ENCODER_PATH']

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)


storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(LABEL_ENCODER_PATH)
blob.download_to_filename("/tmp/label_encoder_obj.pkl")
with open("/tmp/label_encoder_obj.pkl", "rb") as f:
    label_encoders = pickle.load(f)


class PredictionInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

def preprocess_input(data: List[PredictionInput]) -> List[List[float]]:
    df = pd.DataFrame([d.dict() for d in data])
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])
    feature_order = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']
    return df[feature_order].values.tolist()

@app.post("/predict")
async def predict(inputs: List[PredictionInput]):
    try:
        instances = preprocess_input(inputs)
        response = endpoint.predict(instances=instances)
        predictions = ["Yes" if pred >= 0.5 else "No" for pred in response.predictions]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        instances = preprocess_input([PredictionInput(**row) for row in df.to_dict(orient="records")])
        response = endpoint.predict(instances=instances)
        predictions = ["Yes" if pred >= 0.5 else "No" for pred in response.predictions]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

