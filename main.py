from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from models.fraud_detector import FraudDetector
from sklearn.metrics import confusion_matrix, precision_score, recall_score

app = FastAPI(title="Fraud Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detector
fraud_detector = FraudDetector()
model_loaded = fraud_detector.load_model()

# Data Models
class Transaction(BaseModel):
    transaction_id: str
    amount: float
    timestamp: datetime
    merchant_id: str
    customer_id: str
    location: str
    device_id: Optional[str] = None
    payer_id: str
    payee_id: str
    channel: str
    payment_mode: str
    gateway_bank: str
    is_fraud_predicted: bool
    is_fraud_reported: bool

class FraudReport(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_level: str
    detection_method: str
    timestamp: datetime

# In-memory storage (replace with database in production)
transactions = []
fraud_reports = []

# Sample data generation
def generate_sample_data():
    channels = ["Mobile", "Web", "ATM", "POS"]
    payment_modes = ["Credit Card", "Debit Card", "UPI", "Net Banking"]
    banks = ["Bank A", "Bank B", "Bank C", "Bank D"]
    
    for i in range(1000):
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
        is_fraud_pred = np.random.random() < 0.1
        is_fraud_reported = is_fraud_pred if np.random.random() < 0.8 else not is_fraud_pred
        
        transaction = Transaction(
            transaction_id=f"TXN{i+1:06d}",
            timestamp=timestamp,
            amount=np.random.uniform(100, 10000),
            merchant_id=f"MERCHANT{np.random.randint(1, 101):03d}",
            customer_id=f"CUSTOMER{np.random.randint(1, 101):03d}",
            location=f"LOCATION{np.random.randint(1, 101):03d}",
            device_id=f"DEVICE{np.random.randint(1, 101):03d}" if np.random.random() < 0.5 else None,
            payer_id=f"PAYER{np.random.randint(1, 101):03d}",
            payee_id=f"PAYEE{np.random.randint(1, 101):03d}",
            channel=np.random.choice(channels),
            payment_mode=np.random.choice(payment_modes),
            gateway_bank=np.random.choice(banks),
            is_fraud_predicted=is_fraud_pred,
            is_fraud_reported=is_fraud_reported
        )
        transactions.append(transaction)

# Generate sample data on startup
generate_sample_data()

@app.get("/")
async def root():
    return {
        "message": "Welcome to Fraud Detection API",
        "version": "1.0.0",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "endpoints": {
            "transactions": "/api/transactions",
            "fraud_reports": "/api/fraud-reports",
            "analyze": "/api/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_status": "loaded" if model_loaded else "not_loaded"}

@app.post("/api/transactions")
async def create_transaction(transaction: Transaction):
    transactions.append(transaction)
    return {"message": "Transaction recorded", "transaction_id": transaction.transaction_id}

@app.get("/api/transactions")
async def get_transactions():
    return transactions

@app.get("/api/transactions/{transaction_id}")
async def get_transaction(transaction_id: str):
    transaction = next((t for t in transactions if t.transaction_id == transaction_id), None)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction

@app.post("/api/analyze")
async def analyze_transaction(transaction: Transaction):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded")
    
    # Prepare transaction data for prediction
    # Note: In a real application, you would need to transform the transaction data
    # to match the model's expected input format
    transaction_data = np.array([[transaction.amount] + [0] * 28])  # Placeholder for V1-V28
    
    # Get prediction from the model
    prediction = fraud_detector.predict(transaction_data)
    
    # Determine risk level based on fraud probability
    risk_level = "LOW"
    if prediction['fraud_probability'] > 0.8:
        risk_level = "HIGH"
    elif prediction['fraud_probability'] > 0.5:
        risk_level = "MEDIUM"
    
    # Create fraud report
    report = FraudReport(
        transaction_id=transaction.transaction_id,
        fraud_score=prediction['fraud_probability'],
        risk_level=risk_level,
        detection_method="ml_model",
        timestamp=datetime.now()
    )
    
    fraud_reports.append(report)
    return report

@app.get("/api/fraud-reports")
async def get_fraud_reports():
    return fraud_reports

@app.get("/api/fraud-reports/{transaction_id}")
async def get_fraud_report(transaction_id: str):
    report = next((r for r in fraud_reports if r.transaction_id == transaction_id), None)
    if not report:
        raise HTTPException(status_code=404, detail="Fraud report not found")
    return report

@app.post("/api/upload-transactions")
async def upload_transactions(file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.StringIO(contents.decode('utf-8')))
        
        # Process transactions
        for _, row in df.iterrows():
            transaction = Transaction(
                transaction_id=str(row.get('transaction_id', '')),
                amount=float(row.get('amount', 0)),
                timestamp=datetime.fromisoformat(row.get('timestamp', datetime.now().isoformat())),
                merchant_id=str(row.get('merchant_id', '')),
                customer_id=str(row.get('customer_id', '')),
                location=str(row.get('location', '')),
                device_id=str(row.get('device_id', '')) if 'device_id' in row else None,
                payer_id=str(row.get('payer_id', '')),
                payee_id=str(row.get('payee_id', '')),
                channel=str(row.get('channel', '')),
                payment_mode=str(row.get('payment_mode', '')),
                gateway_bank=str(row.get('gateway_bank', '')),
                is_fraud_predicted=bool(row.get('is_fraud_predicted', False)),
                is_fraud_reported=bool(row.get('is_fraud_reported', False))
            )
            transactions.append(transaction)
        
        return {"message": f"Successfully processed {len(df)} transactions"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    if not transactions:
        return {"message": "No transactions available"}
    
    amounts = [t.amount for t in transactions]
    return {
        "total_transactions": len(transactions),
        "total_amount": sum(amounts),
        "average_amount": np.mean(amounts),
        "max_amount": max(amounts),
        "min_amount": min(amounts),
        "fraud_reports_count": len(fraud_reports)
    }

@app.get("/transactions")
async def get_transactions_filtered(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    payer_id: Optional[str] = None,
    payee_id: Optional[str] = None,
    transaction_id: Optional[str] = None
):
    filtered_txns = transactions.copy()
    
    if start_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp >= start_date]
    if end_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp <= end_date]
    if payer_id:
        filtered_txns = [t for t in filtered_txns if t.payer_id == payer_id]
    if payee_id:
        filtered_txns = [t for t in filtered_txns if t.payee_id == payee_id]
    if transaction_id:
        filtered_txns = [t for t in filtered_txns if t.transaction_id == transaction_id]
    
    return filtered_txns

@app.get("/fraud-comparison")
async def get_fraud_comparison(
    dimension: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    filtered_txns = transactions.copy()
    
    if start_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp >= start_date]
    if end_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp <= end_date]
    
    df = pd.DataFrame([t.dict() for t in filtered_txns])
    
    if dimension not in ["channel", "payment_mode", "gateway_bank", "payer_id", "payee_id"]:
        raise HTTPException(status_code=400, detail="Invalid dimension")
    
    comparison = df.groupby(dimension).agg({
        'is_fraud_predicted': 'sum',
        'is_fraud_reported': 'sum'
    }).reset_index()
    
    return comparison.to_dict(orient='records')

@app.get("/fraud-trends")
async def get_fraud_trends(
    start_date: datetime,
    end_date: datetime,
    granularity: str = "day"
):
    filtered_txns = [t for t in transactions if start_date <= t.timestamp <= end_date]
    df = pd.DataFrame([t.dict() for t in filtered_txns])
    
    if granularity not in ["hour", "day", "week", "month"]:
        raise HTTPException(status_code=400, detail="Invalid granularity")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if granularity == "hour":
        df['period'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')
    elif granularity == "day":
        df['period'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    elif granularity == "week":
        df['period'] = df['timestamp'].dt.strftime('%Y-%W')
    else:  # month
        df['period'] = df['timestamp'].dt.strftime('%Y-%m')
    
    trends = df.groupby('period').agg({
        'is_fraud_predicted': 'sum',
        'is_fraud_reported': 'sum'
    }).reset_index()
    
    return trends.to_dict(orient='records')

@app.get("/model-evaluation")
async def get_model_evaluation(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    filtered_txns = transactions.copy()
    
    if start_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp >= start_date]
    if end_date:
        filtered_txns = [t for t in filtered_txns if t.timestamp <= end_date]
    
    y_true = [t.is_fraud_reported for t in filtered_txns]
    y_pred = [t.is_fraud_predicted for t in filtered_txns]
    
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return {
        "confusion_matrix": conf_matrix,
        "precision": precision,
        "recall": recall
    } 