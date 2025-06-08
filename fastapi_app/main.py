from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: int 
    balance : float 
    job : str 

class PredictResponse(BaseModel):
    prediction : str 
    confidence : float 


app = FastAPI(
    title='Client Conversion Prediction API',
    version='1.0',
    description="API for predicting term deposit client conversion"
)


@app.get('/', tags=['Info'])
async def root():
    return {'Message':'Welcome to the Client Conversion Prediction API'}


@app.get('/health', tags=['Health'])
async def health_check():
    return {'status':'ok'}

@app.post('/predict', response_model=PredictResponse, tags=['Inference'])
async def predict(request : PredictRequest):
    if request.age < 30:
        pred = "No"
        conf = 0.6
    else:
        pred = "Yes"
        conf = 0.8

    return PredictResponse(prediction=pred, confidence=conf)
