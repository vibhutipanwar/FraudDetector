import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_path = 'models/fraud_model.joblib'
        self.scaler_path = 'models/scaler.joblib'

    def load_data(self, file_path):
        """Load and preprocess the transactions dataset"""
        df = pd.read_csv(file_path)
        
        # Create feature matrix X
        X = pd.DataFrame()
        X['amount'] = df['transaction_amount']
        
        # Create some derived features
        X['hour'] = pd.to_datetime(df['transaction_date']).dt.hour
        X['day_of_week'] = pd.to_datetime(df['transaction_date']).dt.dayofweek
        
        # Add channel frequency features
        X['channel_frequency'] = df.groupby('transaction_channel')['transaction_amount'].transform('count')
        
        # Fill any missing values
        X = X.fillna(0)
        
        # Target variable
        y = df['is_fraud']
        
        return X, y

    def train(self, X, y):
        """Train the fraud detection model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Save the model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        return self.model, self.scaler

    def load_model(self):
        """Load the trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict(self, transaction_data):
        """Predict if a transaction is fraudulent"""
        # Scale the input data
        scaled_data = self.scaler.transform(transaction_data)
        
        # Get prediction and probability
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        return {
            'is_fraudulent': bool(prediction[0]),
            'fraud_probability': float(probability[0][1]),
            'confidence_score': float(max(probability[0]))
        }

    def get_feature_importance(self):
        """Get feature importance scores"""
        feature_names = ['amount', 'hour', 'day_of_week', 'channel_frequency']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False) 
