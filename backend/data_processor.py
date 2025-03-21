"""
Data processing utilities for the FDAM system.
Handles data transformation, feature engineering, and preprocessing.
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import logging
from .config import config

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles all data processing operations for the fraud detection system.
    """
    
    # Feature sets for different model types
    BASIC_FEATURES = [
        'amount', 'currency_code', 'payer_account_age_days', 
        'payer_country_code', 'payee_country_code', 'payment_method_type',
        'is_international', 'is_first_transaction', 'hour_of_day', 
        'day_of_week', 'is_weekend'
    ]
    
    ADVANCED_FEATURES = BASIC_FEATURES + [
        'transaction_velocity_1h', 'transaction_velocity_24h',
        'amount_vs_average', 'distance_from_usual_location',
        'payee_risk_score', 'payer_risk_score', 'device_risk_score'
    ]
    
    def __init__(self):
        """Initialize the data processor with configuration."""
        self.feature_set = config.get('ml.feature_set', 'advanced')
        self.features = self.ADVANCED_FEATURES if self.feature_set == 'advanced' else self.BASIC_FEATURES
        self.categorical_features = [
            'currency_code', 'payer_country_code', 'payee_country_code', 
            'payment_method_type', 'is_international', 'is_weekend', 
            'is_first_transaction'
        ]
        self.numerical_features = [f for f in self.features if f not in self.categorical_features]
        
        # Load any required preprocessing data
        self._load_preprocessing_data()
    
    def _load_preprocessing_data(self):
        """Load necessary preprocessing data like encoders, scalers, etc."""
        # This would load saved encoders, scalers from files or database
        # For this implementation, we'll use simple mappings
        self.country_risk_scores = {
            'US': 0.2, 'CA': 0.3, 'GB': 0.25, 'DE': 0.3, 
            'FR': 0.35, 'JP': 0.2, 'AU': 0.25, 'NZ': 0.25,
            'SG': 0.4, 'HK': 0.45, 'CN': 0.7, 'RU': 0.75,
            'NG': 0.8, 'BR': 0.6, 'MX': 0.65, 'IN': 0.55
        }
        
        self.payment_method_risk_scores = {
            'credit_card': 0.4,
            'debit_card': 0.3,
            'bank_transfer': 0.2,
            'digital_wallet': 0.5,
            'cryptocurrency': 0.8,
            'gift_card': 0.7,
            'prepaid_card': 0.6
        }
        
        # Default values for missing data
        self.default_values = {
            'payer_account_age_days': 0,
            'transaction_velocity_1h': 0,
            'transaction_velocity_24h': 0,
            'amount_vs_average': 1.0,
            'distance_from_usual_location': 0,
            'payer_country_code': 'unknown',
            'payee_country_code': 'unknown',
            'payment_method_type': 'unknown'
        }
    
    def preprocess_transaction(self, transaction: Dict[str, Any], historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Preprocess a single transaction for prediction.
        
        Args:
            transaction: The transaction data
            historical_data: Historical data for the payer/payee if available
            
        Returns:
            Processed features ready for model input
        """
        try:
            processed = {}
            
            # Basic features
            processed['amount'] = float(transaction.get('amount', 0))
            processed['currency_code'] = transaction.get('currency', 'USD')
            
            # Payer features
            payer = transaction.get('payer', {})
            processed['payer_account_age_days'] = payer.get('account_age_days', self.default_values['payer_account_age_days'])
            processed['payer_country_code'] = payer.get('country', self.default_values['payer_country_code'])
            processed['payer_risk_score'] = self.country_risk_scores.get(
                processed['payer_country_code'], 0.5
            )
            
            # Payee features
            payee = transaction.get('payee', {})
            processed['payee_country_code'] = payee.get('country', self.default_values['payee_country_code'])
            processed['payee_risk_score'] = self.country_risk_scores.get(
                processed['payee_country_code'], 0.5
            )
            
            # Payment method
            payment_method = transaction.get('payment_method', {})
            processed['payment_method_type'] = payment_method.get('type', self.default_values['payment_method_type'])
            processed['payment_method_risk'] = self.payment_method_risk_scores.get(
                processed['payment_method_type'], 0.5
            )
            
            # Derived features
            processed['is_international'] = int(processed['payer_country_code'] != processed['payee_country_code'])
            
            # Time-based features
            timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
            processed['hour_of_day'] = timestamp.hour
            processed['day_of_week'] = timestamp.weekday()
            processed['is_weekend'] = int(timestamp.weekday() >= 5)  # 5,6 = Saturday, Sunday
            
            # Use historical data if available for advanced features
            if historical_data and self.feature_set == 'advanced':
                processed['is_first_transaction'] = int(historical_data.get('transaction_count', 0) == 0)
                processed['transaction_velocity_1h'] = historical_data.get('tx_count_1h', self.default_values['transaction_velocity_1h'])
                processed['transaction_velocity_24h'] = historical_data.get('tx_count_24h', self.default_values['transaction_velocity_24h'])
                processed['amount_vs_average'] = processed['amount'] / historical_data.get('avg_amount', processed['amount']) if historical_data.get('avg_amount') else 1.0
                processed['distance_from_usual_location'] = historical_data.get('location_distance', self.default_values['distance_from_usual_location'])
                processed['device_risk_score'] = historical_data.get('device_risk', 0.5)
            else:
                # Default values if no historical data
                processed['is_first_transaction'] = 1
                processed['transaction_velocity_1h'] = self.default_values['transaction_velocity_1h']
                processed['transaction_velocity_24h'] = self.default_values['transaction_velocity_24h']
                processed['amount_vs_average'] = self.default_values['amount_vs_average']
                processed['distance_from_usual_location'] = self.default_values['distance_from_usual_location']
                processed['device_risk_score'] = 0.5
                
            # Filter to include only the required features
            final_features = {k: v for k, v in processed.items() if k in self.features}
            
            # Generate a feature hash for caching
            feature_hash = self._generate_feature_hash(final_features)
            final_features['feature_hash'] = feature_hash
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error preprocessing transaction: {str(e)}")
            # Return empty feature set with feature hash
            return {'feature_hash': self._generate_feature_hash({}), 'error': str(e)}
    
    def batch_preprocess(self, transactions: List[Dict[str, Any]], historical_data: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of transactions.
        
        Args:
            transactions: List of transaction data
            historical_data: Dict mapping payer_id to historical data
            
        Returns:
            List of processed features
        """
        processed_transactions = []
        
        for tx in transactions:
            payer_id = tx.get('payer', {}).get('id')
            payer_history = historical_data.get(payer_id, {}) if historical_data else {}
            processed_tx = self.preprocess_transaction(tx, payer_history)
            processed_transactions.append(processed_tx)
            
        return processed_transactions
    
    def create_model_input(self, processed_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert processed features to model input format.
        
        Args:
            processed_features: Dict of processed features
            
        Returns:
            Feature array ready for model input
        """
        # This would normally one-hot encode categorical features
        # For this implementation, we'll just create a list of values
        feature_array = []
        
        for feature in self.features:
            if feature in processed_features:
                feature_array.append(processed_features[feature])
            else:
                # Use default value if feature is missing
                feature_array.append(self.default_values.get(feature, 0))
        
        return np.array([feature_array])
    
    def post_process_prediction(self, prediction: float, processed_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process model prediction to create the final fraud analysis.
        
        Args:
            prediction: Raw model prediction
            processed_features: The processed features used for prediction
            
        Returns:
            Dict with fraud analysis results
        """
        # Calculate confidence based on prediction value
        if 0.0 <= prediction <= 1.0:
            confidence = max(min(abs(prediction - 0.5) * 2, 1.0), 0.0)
        else:
            confidence = 0.5  # Default for unexpected values
            
        is_fraud = bool(prediction >= config.get('ml.fraud_threshold', 0.6))
        
        # Generate explanation factors
        explanation_factors = self._generate_explanation_factors(processed_features, prediction)
        
        return {
            "score": float(prediction),
            "is_fraud": is_fraud,
            "ml_confidence": float(confidence),
            "explanation_factors": explanation_factors
        }
    
    def _generate_explanation_factors(self, features: Dict[str, Any], prediction: float) -> List[Dict[str, Any]]:
        """
        Generate explanation factors for the prediction.
        
        Args:
            features: Processed features
            prediction: Model prediction
            
        Returns:
            List of factors that influenced the prediction
        """
        factors = []
        
        # Check amount
        if features.get('amount', 0) > 1000:
            factors.append({
                "factor": "HIGH_AMOUNT",
                "description": "Transaction amount is unusually high",
                "contribution": 0.3
            })
            
        # Check if international
        if features.get('is_international', 0) == 1:
            factors.append({
                "factor": "INTERNATIONAL_TRANSACTION",
                "description": "Transaction crosses international borders",
                "contribution": 0.2
            })
            
        # Check velocity
        if features.get('transaction_velocity_1h', 0) > 5:
            factors.append({
                "factor": "HIGH_VELOCITY",
                "description": "Multiple transactions in short time period",
                "contribution": 0.25
            })
            
        # Check unusual amount
        if features.get('amount_vs_average', 1.0) > 3.0:
            factors.append({
                "factor": "UNUSUAL_AMOUNT",
                "description": "Amount significantly higher than user average",
                "contribution": 0.15
            })
            
        # Check risky country
        payer_country = features.get('payer_country_code', 'unknown')
        if self.country_risk_scores.get(payer_country, 0.5) > 0.6:
            factors.append({
                "factor": "HIGH_RISK_COUNTRY",
                "description": f"Transaction from high-risk region ({payer_country})",
                "contribution": 0.2
            })
            
        # If no specific factors, add a generic one
        if not factors:
            if prediction >= 0.6:
                factors.append({
                    "factor": "COMBINED_RISK_FACTORS",
                    "description": "Multiple small risk indicators combined",
                    "contribution": 0.5
                })
            else:
                factors.append({
                    "factor": "LOW_RISK_PROFILE",
                    "description": "Transaction matches normal patterns",
                    "contribution": 0.1
                })
                
        return factors
    
    def _generate_feature_hash(self, features: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the feature set for caching purposes.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Hash string
        """
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()


# Initialize the data processor
data_processor = DataProcessor()
