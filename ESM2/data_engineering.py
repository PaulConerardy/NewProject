import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

class TransactionPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess(self, transactions_df, customers_df):
        # Merge customer information
        df = transactions_df.merge(customers_df, on='customer_id', how='left')
        
        # Convert transaction_date to datetime if not already
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Add temporal features
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df

class FeatureEngineer:
    def __init__(self, window_sizes=[1, 7, 30]):
        self.window_sizes = window_sizes
    
    def _calculate_customer_stats(self, df):
        # Calculate customer-level statistics
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'max', 'min', 'count'],
            'transaction_type': lambda x: x.value_counts().index[0]  # Most common type
        })
        customer_stats.columns = ['avg_amount', 'std_amount', 'max_amount', 
                                'min_amount', 'transaction_count', 'preferred_type']
        return customer_stats
    
    def _calculate_temporal_features(self, df):
        # Calculate temporal features for each customer
        temporal_features = []
        
        for window in self.window_sizes:
            # Sort by date for rolling calculations
            df_sorted = df.sort_values('transaction_date')
            
            # Calculate rolling statistics
            window_stats = df_sorted.groupby('customer_id').rolling(
                window=f'{window}D',
                on='transaction_date'
            )['amount'].agg(['mean', 'std', 'count']).reset_index()
            
            # Rename columns
            window_stats.columns = ['customer_id', 'date_index', 
                                  f'amount_mean_{window}d', 
                                  f'amount_std_{window}d',
                                  f'transaction_count_{window}d']
            
            temporal_features.append(window_stats)
        
        # Merge all temporal features
        temporal_df = temporal_features[0]
        for df in temporal_features[1:]:
            temporal_df = temporal_df.merge(df, on=['customer_id', 'date_index'])
        
        return temporal_df
    
    def _calculate_network_features(self, df):
        # Calculate network-based features
        # For each customer, look at their transaction patterns
        network_features = df.groupby('customer_id').agg({
            'transaction_type': 'nunique',  # Number of different transaction types
            'amount': lambda x: (x > x.mean() + 2*x.std()).sum(),  # Count of unusual amounts
            'transaction_date': lambda x: x.diff().dt.total_seconds().mean()  # Average time between transactions
        })
        
        network_features.columns = ['n_transaction_types', 'n_unusual_amounts', 'avg_time_between_trans']
        return network_features
    
    def engineer_features(self, df):
        # Calculate all feature sets
        customer_stats = self._calculate_customer_stats(df)
        temporal_features = self._calculate_temporal_features(df)
        network_features = self._calculate_network_features(df)
        
        # Merge all features
        features = df.merge(customer_stats, on='customer_id', how='left')
        features = features.merge(temporal_features, on=['customer_id'], how='left')
        features = features.merge(network_features, on='customer_id', how='left')
        
        # Add derived features
        features['amount_to_avg_ratio'] = features['amount'] / features['avg_amount']
        features['transaction_frequency'] = features['transaction_count'] / \
                                          (features['account_age'] + 1)
        
        # Convert categorical variables
        features = pd.get_dummies(features, columns=['business_type', 
                                                   'risk_score', 
                                                   'transaction_type',
                                                   'preferred_type'])
        
        return features

class DataPipeline:
    def __init__(self):
        self.preprocessor = TransactionPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def prepare_data(self, transactions_df, customers_df):
        # Preprocess the data
        processed_df = self.preprocessor.preprocess(transactions_df, customers_df)
        
        # Engineer features
        feature_df = self.feature_engineer.engineer_features(processed_df)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        return feature_df