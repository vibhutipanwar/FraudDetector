"""
Machine learning model for fraud detection.
Handles model loading, prediction, and training.
"""
import logging
import asyncio
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from .config import config
from .database import (
    get_transaction_repository,
    get_model_metrics_repository
)
from .models import TransactionData, FraudPrediction, ModelMetrics

logger = logging.getLogger(__name__)

# Global model instance
_model = None
_model_version = None
_feature_names = None
_scaler = None

async def initialize_model():
    """
    Initialize the ML model by loading from disk or training a new one.
    """
    global _model, _model_version, _feature_names, _scaler
    
    model_path = config.get("ml.model_path", "models/fraud_model.pkl")
    
    try:
        # Try to load existing model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        _model = model_data.get('model')
        _model_version = model_data.get('version')
        _feature_names = model_data.get('feature_names')
        _scaler = model_data.get('scaler')
        
        logger.info(f"Loaded ML model version {_model_version} from {model_path}")
        
    except (FileNotFoundError, pickle.PickleError) as e:
        logger.warning(f"Could not load model from {model_path}: {str(e)}")
        logger.info("Will train new model")
        
        # Train new model
        await train_model()

async def shutdown_model():
    """
    Clean up model resources.
    """
    global _model, _model_version, _feature_names, _scaler
    
    logger.info("Shutting down ML model")
    
    # Release resources
    _model = None
    _model_version = None
    _feature_names = None
    _scaler = None

async def predict_fraud(transaction: TransactionData) -> FraudPrediction:
    """
    Predict whether a transaction is fraudulent.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Fraud prediction result
    """
    global _model, _model_version, _feature_names, _scaler
    
    if _model is None:
        logger.error("Model not initialized")
        raise RuntimeError("Model not initialized")
        
    # Extract features
    features = await extract_features(transaction)
    
    # Convert to numpy array
    feature_array = np.array([features[feature] for feature in _feature_names]).reshape(1, -1)
    
    # Apply scaling
    if _scaler:
        feature_array = _scaler.transform(feature_array)
    
    # Make prediction
    fraud_probability = _model.predict_proba(feature_array)[0, 1]
    is_fraud = fraud_probability > config.get("ml.fraud_threshold", 0.7)
    
    # Determine fraud category if fraud is detected
    fraud_category = None
    if is_fraud:
        fraud_category = determine_fraud_category(features, fraud_probability)
    
    # Create prediction result
    prediction = FraudPrediction(
        transaction_id=transaction.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=float(fraud_probability),
        fraud_category=fraud_category,
        model_version=_model_version,
        prediction_time=datetime.utcnow().isoformat(),
        features_used=_feature_names
    )
    
    return prediction

def determine_fraud_category(features: Dict[str, float], probability: float) -> str:
    """
    Determine the category of fraud based on features and probability.
    
    Args:
        features: Transaction features
        probability: Fraud probability
        
    Returns:
        Fraud category
    """
    # Basic category determination based on features
    # In a real system, this could be more sophisticated
    
    if features.get('is_new_payee', 0) == 1 and features.get('amount_to_avg_ratio', 0) > 3:
        return "unusual_payee"
    elif features.get('high_risk_country', 0) == 1:
        return "international_fraud"
    elif features.get('is_unusual_time', 0) == 1:
        return "account_takeover"
    elif features.get('velocity_alert', 0) == 1:
        return "velocity_abuse"
    elif probability > 0.9:
        return "high_confidence_fraud"
    else:
        return "suspicious_activity"

async def extract_features(transaction: TransactionData) -> Dict[str, float]:
    """
    Extract features from a transaction for model prediction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Dictionary of features
    """
    # Get transaction repository for historical data
    tx_repo = get_transaction_repository()
    
    # Get historical data for the user
    historical_data = await tx_repo.get_historical_data(transaction.payer.id)
    
    # Base features from transaction
    features = {
        'amount': transaction.amount,
        'is_international': 1 if transaction.payee.country != transaction.payer.country else 0,
    }
    
    # Time-based features
    tx_time = datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00'))
    tx_hour = tx_time.hour
    
    features['is_weekend'] = 1 if tx_time.weekday() >= 5 else 0
    features['is_night'] = 1 if tx_hour < 6 or tx_hour >= 22 else 0
    features['is_unusual_time'] = determine_unusual_time(tx_time, historical_data)
    
    # Amount-based features
    avg_amount = historical_data.get('avg_amount', transaction.amount)
    features['amount_to_avg_ratio'] = transaction.amount / avg_amount if avg_amount > 0 else 1.0
    features['is_large_amount'] = 1 if transaction.amount > 1000 else 0
    
    # Velocity-based features
    tx_count_1h = historical_data.get('tx_count_1h', 0)
    tx_count_24h = historical_data.get('tx_count_24h', 0)
    
    features['tx_count_1h'] = tx_count_1h
    features['tx_count_24h'] = tx_count_24h
    features['velocity_alert'] = 1 if tx_count_1h > 3 or tx_count_24h > 10 else 0
    
    # Payee-related features
    features['is_new_payee'] = determine_new_payee(transaction)
    features['high_risk_country'] = determine_high_risk_country(transaction.payee.country)
    
    # Account age features
    account_age_days = historical_data.get('account_age_days', 0)
    features['new_account'] = 1 if account_age_days < 30 else 0
    features['account_age_days'] = account_age_days
    
    # Device and location features
    if transaction.device_info:
        features['is_new_device'] = 1 if transaction.device_info.get('is_new', False) else 0
        features['is_emulator'] = 1 if transaction.device_info.get('is_emulator', False) else 0
    else:
        features['is_new_device'] = 0
        features['is_emulator'] = 0
        
    if transaction.location_info:
        features['location_mismatch'] = 1 if transaction.location_info.get('is_mismatch', False) else 0
    else:
        features['location_mismatch'] = 0
    
    return features

def determine_unusual_time(tx_time: datetime, historical_data: Dict[str, Any]) -> int:
    """
    Determine if transaction time is unusual for this user.
    
    Args:
        tx_time: Transaction time
        historical_data: User's historical data
        
    Returns:
        1 if unusual, 0 otherwise
    """
    # In a real system, this would use historical patterns
    # For now, use a simple heuristic
    tx_hour = tx_time.hour
    
    # Consider 1am-5am as unusual times
    if 1 <= tx_hour <= 5:
        return 1
    
    return 0

def determine_new_payee(transaction: TransactionData) -> int:
    """
    Determine if payee is new for this user.
    
    Args:
        transaction: Transaction data
        
    Returns:
        1 if new payee, 0 otherwise
    """
    # In a real system, this would check against user's payee history
    # For now, use the flag if provided
    return 1 if transaction.payee.get('is_new', False) else 0

def determine_high_risk_country(country_code: str) -> int:
    """
    Determine if country is high risk.
    
    Args:
        country_code: Country code
        
    Returns:
        1 if high risk, 0 otherwise
    """
    high_risk_countries = config.get("ml.high_risk_countries", [])
    return 1 if country_code in high_risk_countries else 0

async def train_model():
    """
    Train a new fraud detection model on historical data.
    """
    global _model, _model_version, _feature_names, _scaler
    
    logger.info("Starting model training")
    
    # Get repositories
    tx_repo = get_transaction_repository()
    metrics_repo = get_model_metrics_repository()
    
    # Get training data timeframe
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=config.get("ml.training_days", 90))
    
    # Get transactions for training
    logger.info(f"Fetching transactions from {start_time} to {end_time}")
    
    # In a real system, this would use a data loader to handle large datasets
    # Simplified approach for demonstration
    query = {
        "timestamp": {
            "$gte": start_time.isoformat(),
            "$lte": end_time.isoformat()
        }
    }
    
    # Fetch transactions (using a hypothetical method)
    # This would need to be implemented based on your data access patterns
    transactions = await fetch_training_transactions(start_time, end_time)
    
    if not transactions:
        logger.error("No transactions found for training")
        return
        
    logger.info(f"Fetched {len(transactions)} transactions for training")
    
    # Process transactions into features and labels
    X, y, feature_names = process_transactions_for_training(transactions)
    
    if len(X) == 0:
        logger.error("No valid training data after processing")
        return
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model pipeline with preprocessing
    logger.info("Training new model")
    
    scaler = StandardScaler()
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Fit scaler
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Generate predictions on test set
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "timestamp": datetime.utcnow().isoformat(),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "model_type": "GradientBoostingClassifier"
    }
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])
    
    # Record metrics in database
    metrics["model_version"] = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    await metrics_repo.record_metrics(metrics)
    
    # Update global model
    _model = model
    _model_version = metrics["model_version"]
    _feature_names = feature_names
    _scaler = scaler
    
    # Save model to disk
    model_path = config.get("ml.model_path", "models/fraud_model.pkl")
    
    model_data = {
        'model': model,
        'version': _model_version,
        'feature_names': feature_names,
        'scaler': scaler,
        'training_date': datetime.utcnow().isoformat(),
        'metrics': metrics
    }
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
    
    logger.info(f"Model training completed. Version: {_model_version}")
    logger.info(f"Model metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")

async def fetch_training_transactions(start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
    """
    Fetch transactions for model training.
    
    Args:
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        List of transactions with fraud labels
    """
    # This is a placeholder - in a real system, this would query the database
    # or a data warehouse for labeled transaction data
    
    tx_repo = get_transaction_repository()
    
    # In a real implementation, you'd use a specialized query for training data
    # This is a simplified example
    
    # Fetch transactions in batches to avoid memory issues
    batch_size = 1000
    all_transactions = []
    
    # This is a simplified approach - in reality you would implement
    # a proper pagination mechanism
    skip = 0
    while True:
        transactions_batch = await tx_repo.collection.find({
            "timestamp": {
                "$gte": start_time.isoformat(),
                "$lte": end_time.isoformat()
            },
            "fraud_analysis.is_labeled": True  # Only get labeled transactions
        }).skip(skip).limit(batch_size).to_list(length=batch_size)
        
        if not transactions_batch:
            break
            
        all_transactions.extend(transactions_batch)
        skip += batch_size
        
        if len(transactions_batch) < batch_size:
            break
    
    return all_transactions

def process_transactions_for_training(transactions: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Process transactions into features and labels for model training.
    
    Args:
        transactions: List of transaction data
        
    Returns:
        Features, labels, and feature names
    """
    if not transactions:
        return np.array([]), np.array([]), []
    
    # Extract features from each transaction
    features_list = []
    labels = []
    
    for tx in transactions:
        # Skip transactions without fraud analysis
        if 'fraud_analysis' not in tx or 'is_fraud' not in tx['fraud_analysis']:
            continue
            
        # Extract basic features
        features = {
            'amount': tx.get('amount', 0),
            'is_international': 1 if tx.get('payee', {}).get('country') != tx.get('payer', {}).get('country') else 0,
        }
        
        # Time-based features
        if 'timestamp' in tx:
            tx_time = datetime.fromisoformat(tx['timestamp'].replace('Z', '+00:00'))
            tx_hour = tx_time.hour
            
            features['is_weekend'] = 1 if tx_time.weekday() >= 5 else 0
            features['is_night'] = 1 if tx_hour < 6 or tx_hour >= 22 else 0
        else:
            features['is_weekend'] = 0
            features['is_night'] = 0
        
        # Device and location features
        if 'device_info' in tx:
            features['is_new_device'] = 1 if tx['device_info'].get('is_new', False) else 0
            features['is_emulator'] = 1 if tx['device_info'].get('is_emulator', False) else 0
        else:
            features['is_new_device'] = 0
            features['is_emulator'] = 0
            
        if 'location_info' in tx:
            features['location_mismatch'] = 1 if tx['location_info'].get('is_mismatch', False) else 0
        else:
            features['location_mismatch'] = 0
            
        # Add any additional features from the transaction
        if 'features' in tx:
            for key, value in tx['features'].items():
                # Only use numeric features
                if isinstance(value, (int, float)):
                    features[key] = value
        
        # Skip transactions with missing essential features
        if 'amount' not in features:
            continue
        
        features_list.append(features)
        labels.append(1 if tx['fraud_analysis']['is_fraud'] else 0)
    
    if not features_list:
        return np.array([]), np.array([]), []
    
    # Create DataFrame for easier processing
    df = pd.DataFrame(features_list)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Get feature names
    feature_names = list(df.columns)
    
    return df.values, np.array(labels), feature_names
