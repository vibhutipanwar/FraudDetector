import sys
from models.fraud_detector import FraudDetector
import os
import pandas as pd

def main():
    try:
        # Print Python environment information
        print("Python version:", sys.version)
        print("Python executable:", sys.executable)
        print("Current working directory:", os.getcwd())
        print("Python path:", sys.path)
        
        # Initialize the fraud detector
        print("\nInitializing FraudDetector...")
        detector = FraudDetector()
        
        # Check if file exists
        file_path = 'transactions_train.csv'
        abs_path = os.path.abspath(file_path)
        print(f"\nChecking if file exists at: {abs_path}")
        if not os.path.exists(file_path):
            print(f"Error: File not found at {abs_path}")
            return
            
        # List directory contents
        print("\nDirectory contents:")
        for item in os.listdir():
            print(f"- {item}")
            
        # Try reading the first few rows to verify the file structure
        print("\nReading first few rows of the dataset...")
        df = pd.read_csv(file_path, nrows=5)
        print("\nDataset columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Load the full dataset
        print("\nLoading full dataset...")
        X, y = detector.load_data(file_path)
        
        # Train the model
        print("\nTraining model...")
        model, scaler = detector.train(X, y)
        
        print("\nModel training completed!")
        print(f"Model saved to: {detector.model_path}")
        print(f"Scaler saved to: {detector.scaler_path}")
        
        # Display feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = detector.get_feature_importance()
        print(feature_importance.head(10))
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 