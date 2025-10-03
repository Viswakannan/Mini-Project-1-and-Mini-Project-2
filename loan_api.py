from fastapi import FastAPI 
from joblib import load 
import numpy as np 

app = FastAPI() 

# Load model 
model = load('loan_approval_model.joblib') 

@app.post("/predict") 
def predict_loan(income: int, loan_amount: int, credit_score: int): 
    input_data = np.array([[income, loan_amount, credit_score]]) 
    prediction = model.predict(input_data)[0] 
    return {"loan_approval": "Approved" if prediction == 1 else "Rejected"}

