import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta

class AdvancedFeatureEngineer:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.label_encoders = {}
        
    def create_advanced_features(self, df):
        """Generate advanced time-series and categorical features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-series pattern features
        df = self._add_temporal_patterns(df)
        df = self._add_seasonal_features(df)
        df = self._add_trend_features(df)
        
        # Categorical encoding features
        df = self._encode_categorical_features(df)
        df = self._create_interaction_features(df)
        
        return df
    
    def _add_temporal_patterns(self, df):
        """Add time-series pattern recognition features"""
        # Group by customer for pattern analysis
        for window in [1, 7, self.lookback_days]:
            # Transaction rhythm features
            df[f'tx_rhythm_{window}d'] = df.groupby('customer_id')['timestamp'].apply(
                lambda x: self._calculate_rhythm_score(x, window)
            ).reset_index(level=0, drop=True)
            
            # Amount pattern features
            df[f'amount_pattern_{window}d'] = df.groupby('customer_id')['amount'].apply(
                lambda x: self._detect_amount_patterns(x, window)
            ).reset_index(level=0, drop=True)
            
            # Behavior change detection
            df[f'behavior_change_{window}d'] = df.groupby('customer_id').apply(
                lambda x: self._detect_behavior_changes(x, window)
            ).reset_index(level=0, drop=True)
        
        return df
    
    def _calculate_rhythm_score(self, timestamps, window):
        """Calculate regularity of transaction timing"""
        if len(timestamps) < 2:
            return 0
            
        # Convert to hours since start
        hours_diff = np.diff(timestamps.astype(np.int64)) / 3.6e12
        
        # Calculate coefficient of variation (lower means more regular)
        if len(hours_diff) > 0:
            return stats.variation(hours_diff, nan_policy='omit')
        return 0
    
    def _detect_amount_patterns(self, amounts, window):
        """Detect patterns in transaction amounts"""
        if len(amounts) < 3:
            return 0
            
        # Calculate autocorrelation
        autocorr = pd.Series(amounts).autocorr(lag=1)
        return autocorr if not pd.isna(autocorr) else 0
    
    def _detect_behavior_changes(self, customer_df, window):
        """Detect sudden changes in transaction behavior"""
        if len(customer_df) < 2:
            return 0
            
        recent_mean = customer_df['amount'].tail(max(2, len(customer_df)//4)).mean()
        historical_mean = customer_df['amount'].head(max(2, len(customer_df)*3//4)).mean()
        
        if historical_mean == 0:
            return 0
        
        return abs(recent_mean - historical_mean) / historical_mean
    
    def _add_seasonal_features(self, df):
        """Add seasonal decomposition features"""
        # Group by customer and resample to daily frequency
        for customer_id in df['customer_id'].unique():
            mask = df['customer_id'] == customer_id
            customer_series = df[mask].set_index('timestamp')['amount'].resample('D').sum().fillna(0)
            
            if len(customer_series) > 2:  # Need at least 3 points for decomposition
                try:
                    decomposition = seasonal_decompose(
                        customer_series, 
                        period=7,  # Weekly seasonality
                        extrapolate_trend=True
                    )
                    
                    # Add decomposition components
                    df.loc[mask, 'seasonal_strength'] = np.std(decomposition.seasonal) / np.std(customer_series)
                    df.loc[mask, 'trend_strength'] = np.std(decomposition.trend) / np.std(customer_series)
                    df.loc[mask, 'residual_strength'] = np.std(decomposition.resid) / np.std(customer_series)
                except:
                    df.loc[mask, 'seasonal_strength'] = 0
                    df.loc[mask, 'trend_strength'] = 0
                    df.loc[mask, 'residual_strength'] = 0
            else:
                df.loc[mask, ['seasonal_strength', 'trend_strength', 'residual_strength']] = 0
                
        return df
    
    def _add_trend_features(self, df):
        """Add trend-based features"""
        for window in [7, self.lookback_days]:
            df[f'amount_trend_{window}d'] = df.groupby('customer_id')['amount'].apply(
                lambda x: self._calculate_trend(x, window)
            ).reset_index(level=0, drop=True)
            
            df[f'frequency_trend_{window}d'] = df.groupby('customer_id')['timestamp'].apply(
                lambda x: self._calculate_frequency_trend(x, window)
            ).reset_index(level=0, drop=True)
        
        return df
    
    def _calculate_trend(self, series, window):
        """Calculate the trend coefficient"""
        if len(series) < 2:
            return 0
            
        try:
            x = np.arange(len(series))
            slope, _, _, _, _ = stats.linregress(x, series)
            return slope
        except:
            return 0
    
    def _calculate_frequency_trend(self, timestamps, window):
        """Calculate trend in transaction frequency"""
        if len(timestamps) < 2:
            return 0
            
        try:
            # Calculate daily transaction counts
            daily_counts = pd.Series(timestamps).dt.date.value_counts().sort_index()
            x = np.arange(len(daily_counts))
            slope, _, _, _, _ = stats.linregress(x, daily_counts)
            return slope
        except:
            return 0
    
    def _encode_categorical_features(self, df):
        """Encode categorical variables with memory of previous encodings"""
        categorical_cols = [
            'transaction_type', 'recipient_country', 'customer_type',
            'country', 'profession', 'naics_code'
        ]
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('MISSING'))
            else:
                # Handle new categories that weren't in training data
                new_categories = set(df[col].unique()) - set(self.label_encoders[col].classes_)
                if new_categories:
                    # Combine old and new categories
                    self.label_encoders[col].classes_ = np.concatenate([
                        self.label_encoders[col].classes_,
                        np.array(list(new_categories))
                    ])
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('MISSING'))
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features between categorical variables"""
        # Transaction type and customer type interaction
        df['tx_customer_interaction'] = df['transaction_type_encoded'] * df['customer_type_encoded']
        
        # Country risk interaction
        df['country_risk_interaction'] = df['country_encoded'] * df['risk_score']
        
        # Professional risk interaction
        if 'profession_encoded' in df.columns:
            df['profession_risk_interaction'] = df['profession_encoded'] * df['risk_score']
        
        # Business risk interaction
        if 'naics_code_encoded' in df.columns:
            df['business_risk_interaction'] = df['naics_code_encoded'] * df['risk_score']
        
        return df