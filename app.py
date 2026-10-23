from fastapi import FastAPI
from pydantic import BaseModel,field
import pandas as pd
import joblib   
import uvicorn
import datetime

#1. load our model and preprocessor
#2. instantiate our fastapi app
#3. define our validation using pydantic base model
#4. write our decorated function to handle the prediction
#4.1. convert to dataframe. 4.2: write the structure/pipeline for engineering new features. 4.3: transform the data using the preprocessor. 4.4: make prediction using the model. 4.5: return the prediction as a response
#4.5: decorate our function with our instantiated fastapi app and the endpoint to handle the prediction

rf_model = joblib.load('notebook/rf_model.joblib')
preprocessor = joblib.load('notebook/preprocessor.joblib')

app = FastAPI(title = "Nova Pay Fraud Detection System API")


 #input data validation using pydantic base model
class TransactionData(BaseModel):
    home_country: str = field(example="us")
    source_currency: str = field(example="usd")
    dest_currency: str = field(example="ngn")
    channel: str = field(example="web")
    amount_src: float = field(example=100.0)
    amount_usd: float = field(example=100.0)
    fee: float = field(example=1.0)
    exchange_rate_src_to_dest: float = field(example=1.0)
    new_device: bool = field(example=0)
    ip_country: str = field(example="us")
    location_mismatch: bool = field(example=0)
    ip_risk_score: float = field(example=0.0)
    kyc_tier: str = field(example="standard")
    account_age_days: int = field (example=100)
    device_trust_score: float = field (example=0.0)
    chargeback_history_count: int = field (example=0)
    chargeback_history_amount: float = field (example=0.0)
    risk_score_internal: float = field (example=0.0)
    risk_score_external: float = field (example=0.0)
    txn_velocity_1h: float = field (example=0.0)
    txn_velocity_24h: float = field (example=0.0)
    corridor_risk: float = field (example=0.0)
    timestamp: datetime = field (example="2026-10-20 00:00:00")

   step 1 #define function to convert the input data to a pandas dataframe
def predict_fraud(transaction_data: TransactionData) -> dict:
    # convert our data from Transaction class to dataframe
    df = pd.DataFrame([transaction.model_dump()])

   step 2 # engineer new features from our incoming data:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['age_bucket'] = pd.cut(df['account_age_days'],
                          bins=[0, 30, 90, 180, 365, 2000],
                          labels=[',30d', '30-90d', '90-180d', '180-365d', '.1yr'])
fraud_by_age=df.groupby('age_bucket')['is_fraud'].mean()
print(fraud_by_age)






app = FastAPI()

class TransactionData(BaseModel):
    # Define the structure of your input data here
    pass

@app.post("/predict")
async def predict(transaction_data: TransactionData):
    # Implement your prediction logic here
    pass
