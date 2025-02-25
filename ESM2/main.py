import pandas as pd
import numpy as np
from data_generation import TransactionGenerator
from data_engineering import DataPipeline
from detection_models import MoneyLaunderingDetector

# Generate synthetic transaction data
print("Generating synthetic transaction data...")
generator = TransactionGenerator(n_customers=5000, n_transactions=50000)
transactions, customers = generator.generate_transactions()

# Prepare features
print("\nPreparing features...")
pipeline = DataPipeline()
feature_df = pipeline.prepare_data(transactions, customers)

# Select features for modeling
feature_columns = [col for col in feature_df.columns 
                  if col not in ['transaction_id', 'customer_id', 'name', 
                                'address', 'transaction_date', 'pattern_type']]
X = feature_df[feature_columns]
y = feature_df['is_suspicious']

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
print("\nTraining and evaluating the model...")
detector = MoneyLaunderingDetector()
detector.train(X_train, y_train)

# Evaluate the model
print("\nModel Evaluation Report:")
# eval_report = detector.evaluate(X_test, y_test)
# print(eval_report)

# Analyze suspicious transactions by pattern type
print("\nAnalyzing suspicious transactions by pattern type:")
suspicious_patterns = feature_df[feature_df['is_suspicious']]['pattern_type'].value_counts()
print("\nDistribution of suspicious patterns:")
print(suspicious_patterns)

# Calculate detection rate by pattern type
print("\nDetection rate by pattern type:")
y_pred = detector.predict(X_test)
test_results = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'pattern_type': feature_df.loc[y_test.index, 'pattern_type']
})

pattern_performance = test_results.groupby('pattern_type').agg({
    'true_label': 'count',
    'predicted_label': lambda x: (x == 1).sum()
}).rename(columns={'predicted_label': 'correct_predictions'})

pattern_performance['detection_rate'] = pattern_performance['correct_predictions'] / pattern_performance['true_label']
print(pattern_performance)