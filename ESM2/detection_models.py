import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.knn = KNN(contamination=0.1)
        self.copod = COPOD(contamination=0.1)
        self.scaler = StandardScaler()
        
    def fit(self, X):
        # Convert datetime columns to numeric
        X_numeric = X.copy()
        datetime_cols = X_numeric.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Fit all models
        self.isolation_forest.fit(X_scaled)
        self.knn.fit(X_scaled)
        self.copod.fit(X_scaled)
        
    def predict(self, X):
        # Convert datetime columns to numeric
        X_numeric = X.copy()
        datetime_cols = X_numeric.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
            
        # Scale the features
        X_scaled = self.scaler.transform(X_numeric)
        
        # Get predictions from all models
        if_pred = self.isolation_forest.predict(X_scaled)
        knn_pred = self.knn.predict(X_scaled)
        copod_pred = self.copod.predict(X_scaled)
        
        # Combine predictions (majority voting)
        predictions = np.vstack([if_pred, knn_pred, copod_pred])
        final_pred = np.apply_along_axis(
            lambda x: -1 if np.sum(x == -1) >= 2 else 1, 
            axis=0, 
            arr=predictions
        )
        
        return final_pred

class SupervisedDetector:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    def fit(self, X, y):
        # Convert datetime columns to numeric
        X_numeric = X.copy()
        datetime_cols = X_numeric.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
        
        self.model.fit(X_numeric, y)
        
    def predict(self, X):
        # Convert datetime columns to numeric
        X_numeric = X.copy()
        datetime_cols = X_numeric.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
            
        return self.model.predict(X_numeric)
    
    def predict_proba(self, X):
        # Convert datetime columns to numeric
        X_numeric = X.copy()
        datetime_cols = X_numeric.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
            
        return self.model.predict_proba(X_numeric)
    
    def get_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

class MoneyLaunderingDetector:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.supervised_detector = SupervisedDetector()
        
    def train(self, X, y=None):
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        # If labels are provided, train supervised model
        if y is not None:
            self.supervised_detector.fit(X, y)
    
    def predict(self, X):
        # Get predictions from both models
        anomaly_pred = self.anomaly_detector.predict(X)
        
        # Convert isolation forest predictions to binary (1 for normal, 0 for anomaly)
        anomaly_pred = (anomaly_pred == 1).astype(int)
        
        # If supervised model is trained, use it as well
        try:
            supervised_pred = self.supervised_detector.predict(X)
            # Combine predictions (more weight to supervised model)
            final_pred = (supervised_pred * 0.7 + anomaly_pred * 0.3 > 0.5).astype(int)
        except:
            final_pred = anomaly_pred
        
        return final_pred
    
    def evaluate(self, X, y_true):
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        # plt.show()
        
        # # Get feature importance if supervised model is trained and has feature names
        # try:
        #     if hasattr(X, 'columns') and hasattr(self.supervised_detector.model, 'feature_importances_'):
        #         feature_importance = self.supervised_detector.get_feature_importance(
        #             X.columns
        #         )
        #         plt.figure(figsize=(12, 6))
        #         sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
        #         plt.title('Top 20 Most Important Features')
        #         plt.show()
        # except Exception as e:
        #     print(f"Note: Could not generate feature importance plot: {str(e)}")
        
        return report