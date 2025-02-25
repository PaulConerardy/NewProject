import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from data.transaction_generator import TransactionGenerator
from features.feature_engineer import TransactionFeatureEngineer
from features.network_features import TransactionNetworkAnalyzer
from models.ml_detector import MLDetector

def test_data_generation():
    """Test transaction data generation"""
    generator = TransactionGenerator(n_customers=1000, n_transactions=10000)
    df = generator.generate_dataset()
    
    # Basic validation
    print("\n=== Data Generation Tests ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nAnomaly distribution:")
    print(df['is_anomaly'].value_counts(normalize=True))
    
    return df

def test_feature_engineering(df):
    """Test feature engineering pipeline"""
    engineer = TransactionFeatureEngineer()
    df_featured, group_stats = engineer.engineer_features(df)
    
    print("\n=== Feature Engineering Tests ===")
    print(f"Number of features: {len(df_featured.columns)}")
    
    # Test for NaN values in engineered features
    nan_cols = df_featured.columns[df_featured.isna().any()].tolist()
    if nan_cols:
        print("\nColumns with NaN values:")
        print(nan_cols)
    
    return df_featured

def visualize_results(df_featured):
    """Create validation visualizations"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature correlations with is_anomaly
    plt.subplot(2, 2, 1)
    # Select only numeric columns for correlation
    numeric_cols = df_featured.select_dtypes(include=[np.number]).columns
    correlations = df_featured[numeric_cols].corr()['is_anomaly'].sort_values()
    correlations[-10:].plot(kind='barh')
    plt.title('Top 10 Features Correlated with Anomalies')
    
    # Plot 2: Anomaly distribution across risk levels
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_featured, x='risk_level', y='amount_peer_dev')  # Changed to amount_peer_dev
    plt.title('Peer Deviation by Risk Level')
    
    # Plot 3: Transaction patterns
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=df_featured.sample(1000),
        x='amount',
        y='amount_velocity_30d',
        hue='is_anomaly',
        alpha=0.6
    )
    plt.title('Transaction Patterns')
    
    # Plot 4: Feature importance for anomaly detection
    plt.subplot(2, 2, 4)
    feature_cols = [col for col in numeric_cols if col.endswith('_dev_30d')]
    avg_values = df_featured[feature_cols].mean().sort_values()
    avg_values.plot(kind='barh')
    plt.title('Average Deviation Features')
    
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/validation_results.png')
    plt.close()

def test_model(df_featured):
    """Test anomaly detection model"""
    print("\n=== Model Testing ===")
    
    # Prepare features
    feature_cols = [
        # Amount-based features
        'amount_zscore_30d', 'amount_peer_dev', 
        'amount_mean_30d', 'amount_std_30d',
        'amount_roundness', 'near_threshold',
        
        # Frequency and velocity features
        'freq_hist_dev_30d', 'amount_velocity_30d',
        'tx_count_30d', 'time_since_last_tx',
        
        # Time-based features
        'outside_business_hours', 'is_weekend', 'is_night',
        
        # Risk features
        'overall_risk_score', 'risk_score', 'structuring_risk'
    ]
    
    # Handle missing or infinite values
    X = df_featured[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Adjust contamination based on actual anomaly rate in data
    actual_anomaly_rate = df_featured['is_anomaly'].mean()
    contamination = max(actual_anomaly_rate, 0.01)  # Set minimum contamination to 1%
    
    # Initialize and train model with adjusted parameters
    detector = MLDetector(contamination=contamination)
    detector.fit(X)
    
    # Get predictions with adjusted threshold
    scores = detector.isolation_forest.score_samples(X)
    pd.DataFrame(scores).to_csv('scores.csv')
    threshold = np.percentile(scores, contamination * 100)
    pred_labels = np.where(scores < threshold, 1, 0)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(df_featured['is_anomaly'], pred_labels))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df_featured['is_anomaly'], pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/confusion_matrix.png')
    plt.close()
    
    # Analyze false positives and negatives
    fp_mask = (df_featured['is_anomaly'] == 0) & (pred_labels == 1)
    fn_mask = (df_featured['is_anomaly'] == 1) & (pred_labels == 0)
    
    print("\nFalse Positive Analysis:")
    print(df_featured[fp_mask][feature_cols].describe())
    
    print("\nFalse Negative Analysis:")
    print(df_featured[fn_mask][feature_cols].describe())
    
    return pred_labels

def main():
    # Test data generation
    df = test_data_generation()
    
    # Test feature engineering
    df_featured = test_feature_engineering(df)
    
    # Test model
    predictions = test_model(df_featured)
    
    # Add predictions to featured dataframe
    df_featured['predicted_anomaly'] = predictions
    
    # Visualize results
    visualize_results(df_featured)
    
    # Save processed data
    output_path = '/Users/paulconerardy/Documents/Trae/Anomaly v2/data/featured_transactions.csv'
    df_featured.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    main()