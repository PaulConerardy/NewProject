from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURE_COLS, RISK_THRESHOLDS  # Changed from data.config to config
from sklearn.preprocessing import RobustScaler  # Changed from StandardScaler

class MLDetector:
    def __init__(self, contamination=0.1):
        # Adjust parameters for better numerical stability
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=500,
            max_samples='auto',
            random_state=42,
            bootstrap=True,
            n_jobs=-1,
            max_features=1.0  # Use all features
        )
        
        # Change to RobustScaler for better handling of outliers
        self.scaler = RobustScaler(quantile_range=(1, 99))
        self.feature_cols = FEATURE_COLS
        self.fitted_features = None  # Add tracking of fitted features

    def fit(self, X):
        """Train the anomaly detection model with improved preprocessing"""
        # Ensure we only use specified features and handle NaN values first
        X = X[self.feature_cols].copy()
        
        # Handle NaN values before any other preprocessing
        X = self._handle_nan_values(X)
        X = self._preprocess_features(X)
        
        # Convert to numpy array to avoid feature names warning
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the isolation forest with numpy array
        self.isolation_forest.fit(X_scaled)
        
        return self

    def predict(self, X):
        """Predict anomalies with improved preprocessing"""
        X = X[self.feature_cols].copy()
        
        # Handle NaN values first
        X = self._handle_nan_values(X)
        X = self._preprocess_features(X)
        
        # Convert to numpy array
        X_scaled = self.scaler.transform(X)
        
        # Get scores
        scores = self.isolation_forest.score_samples(X_scaled)
        scores.to_csv('scores.csv')
        threshold = self._calculate_dynamic_threshold(scores)
        
        return np.where(scores < threshold, -1, 1)

    def _handle_nan_values(self, X):
        """Handle NaN values specifically for each feature type"""
        for column in X.columns:
            if X[column].isnull().any():
                if X[column].dtype in ['float64', 'int64']:
                    # For numeric columns, use rolling median with forward fill
                    X[column] = X[column].fillna(
                        X[column].rolling(window=3, min_periods=1).median()
                    )
                    # If still NaN, use global median
                    X[column] = X[column].fillna(X[column].median())
                else:
                    # For categorical columns, use mode
                    X[column] = X[column].fillna(X[column].mode()[0])
        
        return X
    
    def _apply_rules(self, X):
        """Apply rule-based detection without recursive preprocessing"""
        # Implement basic rule checks without additional preprocessing
        rule_scores = np.zeros(len(X))
        
        # Add rule-based scores directly
        for col in X.columns:
            if col.endswith('_zscore'):
                rule_scores += np.abs(X[col])
            
        return rule_scores / len(X.columns)

    def _preprocess_features(self, X):
        """Preprocess features to handle missing and infinite values"""
        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values per feature type
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_features:
            if X[col].isnull().any():
                # Use median for highly skewed features
                if abs(X[col].skew()) > 1:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mean())
        
        return X
    
    def _calculate_dynamic_threshold(self, scores):
        """Calculate threshold using robust statistics"""
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        # Use IQR-based threshold
        threshold = q1 - 1.5 * iqr
        
        # Ensure minimum contamination
        min_anomalies = int(len(scores) * self.isolation_forest.contamination)
        if (scores < threshold).sum() < min_anomalies:
            threshold = np.percentile(scores, self.isolation_forest.contamination * 100)
            
        return threshold

    def _combine_scores(self, if_scores, rule_scores):
        """Combine different detection methods with adjusted weights"""
        if_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        # Adjust weights to give more importance to rule-based detection
        weights = {
            'isolation_forest': 0.4,  # Reduced from 0.6
            'rules': 0.6             # Increased from 0.4
        }
        
        return (weights['isolation_forest'] * if_normalized + 
                weights['rules'] * rule_scores)