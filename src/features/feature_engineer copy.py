import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class TransactionFeatureEngineer:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
    
    def engineer_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Velocity and frequency features
        df = self._create_velocity_features(df)
        
        # Pattern-based features
        df = self._create_pattern_features(df)
        
        # Network-based features
        # df = self._create_network_features(df)
        
        # Risk-based features
        df = self._create_risk_features(df)
        
        # Add new deviation features
        df = self._create_historical_deviation_features(df)

        # Create and validate peer groups
        df = self._create_peer_groups(df)
        group_stats = self._validate_peer_groups(df)
        # print(df.head())
        # Create peer deviation features
        df, group_stats = self._create_peer_deviation_features(df)
        print(df.head())
        return df, group_stats
        
        # return df
    
    def _create_time_features(self, df):
        """Create time-based features"""
        # Basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        
        # Time windows for aggregations
        df['date'] = df['timestamp'].dt.date
        
        # Business hour deviation
        business_hours = (df['hour'] >= 9) & (df['hour'] <= 17) & (~df['is_weekend'])
        df['outside_business_hours'] = (~business_hours).astype(int)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features"""
        # Set timestamp as index for rolling operations
        df = df.set_index('timestamp').sort_index()
        
        # Rolling statistics per customer
        for window in [1, 7, 30]:
            # Calculate rolling averages and standard deviations
            roll_stats = df.groupby('customer_id')['amount'].rolling(
                window=f'{window}D'
            ).agg(['mean', 'std']).reset_index()
            
            df[f'amount_mean_{window}d'] = roll_stats['mean']
            df[f'amount_std_{window}d'] = roll_stats['std']
            
            # Calculate z-score
            df[f'amount_zscore_{window}d'] = (
                df['amount'] - df[f'amount_mean_{window}d']
            ) / df[f'amount_std_{window}d'].replace(0, 1)
        
        # Reset index to get timestamp back as a column
        df = df.reset_index()
        
        # Round number detection
        df['amount_roundness'] = df['amount'].apply(
            lambda x: len(str(int(x))) - len(str(int(x)).rstrip('0'))
        )
        
        # Just below threshold detection
        df['near_threshold'] = (
            ((df['amount'] > 9000) & (df['amount'] < 10000)) |
            ((df['amount'] > 4500) & (df['amount'] < 5000))
        ).astype(int)
        
        return df
    
    def _create_velocity_features(self, df):
        """Create velocity and frequency-based features"""
        # Set timestamp as index for rolling operations
        df = df.set_index('timestamp').sort_index()
        
        # Transaction frequency
        for window in [1, 7, 30]:
            # Count transactions using count() instead of size()
            freq = df.groupby('customer_id').rolling(
                window=f'{window}D'
            )['amount'].count().reset_index()
            
            df[f'tx_count_{window}d'] = freq['amount']
            
            # Calculate amount velocity
            amount_sum = df.groupby('customer_id')['amount'].rolling(
                window=f'{window}D'
            ).sum().reset_index()
            
            df[f'amount_velocity_{window}d'] = amount_sum['amount'] / window
        
        # Reset index
        df = df.reset_index()
        
        # Time between transactions
        df['time_since_last_tx'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600
        
        return df
    
    def _create_pattern_features(self, df):
        """Create pattern-based features"""
        # Set timestamp as index for rolling operations
        df = df.set_index('timestamp').sort_index()
        
        # Repetitive patterns
        for window in [7, 30]:
            # Same amount patterns
            same_amount = df.groupby('customer_id').rolling(
                window=f'{window}D'
            )['amount'].apply(
                lambda x: (x.nunique() == 1) and (len(x) > 1)
            ).reset_index()
            df[f'same_amount_pattern_{window}d'] = same_amount['amount'].astype(int)
            
            # Same recipient patterns - using string-based approach
            def check_same_recipient(x):
                if len(x) <= 1:
                    return 0
                return int(len(set(x)) == 1)
            
            # recipient_patterns = df.groupby('customer_id').rolling(
            #     window=f'{window}D'
            # ).apply(
            #     lambda x: check_same_recipient(x['recipient_country'].values)
            # ).reset_index()
            # df[f'same_recipient_pattern_{window}d'] = recipient_patterns[0].astype(int)
        
        # Reset index
        df = df.reset_index()
        
        # Structuring patterns
        df['structuring_risk'] = (
            (df['near_threshold'] == 1) &
            (df['time_since_last_tx'] < 48)
        ).astype(int)
        
        return df
    
    def _create_network_features(self, df):
        """Create network-based features"""
        # Set timestamp as index for rolling operations
        df = df.set_index('timestamp').sort_index()
        
        # Country network features - modified approach for categorical data
        country_stats = df.groupby('customer_id').rolling(
            window='30D'
        ).agg({
            'recipient_country': lambda x: x.nunique()
        }).reset_index()
        df['unique_countries_30d'] = country_stats['recipient_country']
        
        # High-risk country ratio - modified approach
        high_risk_countries = ['CN']  # Example high-risk countries
        risk_stats = df.groupby('customer_id').rolling(
            window='30D'
        ).agg({
            'recipient_country': lambda x: (x.isin(high_risk_countries)).mean()
        }).reset_index()
        df['high_risk_country_ratio_30d'] = risk_stats['recipient_country']
        
        # Reset index
        df = df.reset_index()
        
        return df
    
    def _create_risk_features(self, df):
        """Create risk-based features"""
        # Combine multiple risk factors
        df['overall_risk_score'] = (
            df['risk_score'] * 0.3 +
            # df['high_risk_country_ratio_30d'] * 0.2 +
            df['structuring_risk'] * 0.2 +
            df['outside_business_hours'] * 0.1 +
            df['amount_roundness'] * 0.1 +
            df['near_threshold'] * 0.1
        )
        
        # Risk level categories
        df['risk_level'] = pd.qcut(
            df['overall_risk_score'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def _create_historical_deviation_features(self, df):
        """Create features based on customer's historical behavior deviation"""
        # Set timestamp as index for rolling operations
        df_original = df.copy()
        df = df.set_index(['timestamp', 'customer_id']).sort_index()
        
        # Calculate baseline periods
        for window in [7, 30]:
            # Amount deviation from historical average
            hist_amount_stats = df.groupby(level=1)['amount'].rolling(
                window=window, min_periods=1
            ).agg(['mean', 'std'])
            hist_amount_stats.to_csv(f'hist_amount_stats_{window}.csv')
            # Reset index properly to avoid duplicates
            hist_amount_stats = hist_amount_stats.reset_index(level=0)
            
            # Merge back the statistics
            df = df.join(hist_amount_stats, rsuffix='_hist')
            # print(hist_amount_stats.tail())
            # Calculate deviations
            df[f'amount_hist_dev_{window}d'] = (
                df['amount'] - df['mean']
            ) / df['std'].replace(0, 1)
            
            # Drop temporary columns
            df = df.drop(['mean', 'std'], axis=1)
            
            # Transaction frequency deviation
            hist_freq = df.groupby(level=1)['amount'].rolling(
                window=window
            ).count()
            hist_freq = hist_freq.reset_index(level=0)
            hist_freq.name = f'count_{window}d'
            hist_freq = hist_freq.rename(columns= {'amount': f'count_{window}d'})
            
            # Merge frequency stats
            df = df.join(hist_freq, rsuffix='_hist')
            
            # Calculate current frequency (1-day window)
            current_freq = df.groupby(level=1)['amount'].rolling(1).count()
            current_freq = current_freq.reset_index(level=0)
            current_freq.name = f'current_count_{window}d'
            current_freq = current_freq.rename(columns= {'amount': f'current_count_{window}d'})
            
            df = df.join(current_freq, rsuffix='_hist')
            # new_name = f'current_count_{window}d'
            # df = df.rename(columns= {'amount': f'current_count_{window}d'})
            # print('------')
            # print('columns')
            # print(df.columns)
            
            # Calculate frequency deviation
            df[f'freq_hist_dev_{window}d'] = (
                df[f'current_count_{window}d'] - df[f'count_{window}d']
            ) / df[f'count_{window}d'].replace(0, 1)
            
            # Drop temporary columns
            df = df.drop([f'count_{window}d', f'current_count_{window}d'], axis=1)
        
        # Reset index for further processing
        df = df.reset_index(drop=True)
        
        # Copy new features to original dataframe
        new_features = set([col for col in df.columns if col not in df_original.columns])
        # print(df.columns)
        # Remove duplicate columns
        df = df.loc[:,~df.columns.duplicated()].copy()
        # Copy new features while maintaining index alignment
        for feature in new_features:
            # print('df[feature]')
            # print(df[feature].tail())
            df_original[feature] = df[feature].values
        

        return df_original
    
    def _create_peer_groups(self, df):
        """Create sophisticated peer groups based on multiple characteristics"""
        # Calculate transaction behavior metrics
        df['avg_monthly_amount'] = df.groupby('customer_id')['amount'].transform('mean')
        df['tx_frequency'] = df.groupby('customer_id').size() / \
            (df.groupby('customer_id')['timestamp'].transform('max') - 
             df.groupby('customer_id')['timestamp'].transform('min')).dt.days
        
        # Create activity segments with proper handling of missing values
        labels = ['VS', 'S', 'M', 'L', 'VL']
        df['amount_segment'] = pd.qcut(
            df['avg_monthly_amount'],
            q=5,
            labels=labels
        )
        # Handle missing values after creating categories
        df['amount_segment'] = df['amount_segment'].cat.add_categories(['Unknown']).fillna('Unknown')
        
        # Similar approach for frequency segments
        freq_labels = ['Low', 'Medium', 'High']
        df['frequency_segment'] = pd.qcut(
            df['tx_frequency'],
            q=3,
            labels=freq_labels
        )
        df['frequency_segment'] = df['frequency_segment'].cat.add_categories(['Unknown']).fillna('Unknown')
        
        # Define peer groups based on multiple characteristics
        df['peer_group'] = df.apply(self._assign_peer_group, axis=1)
        
        return df
    
    def _assign_peer_group(self, row):
        """Assign peer group based on customer characteristics and behavior"""
        base_group = []
        
        if row['customer_type'] == 'individual':
            # Individual peer grouping
            base_group.extend([
                str(row['profession']) if pd.notna(row['profession']) else 'unknown_profession',
                str(row['amount_segment']),
                # str(row['frequency_segment']),
                # 'domestic' if row['country'] == row['recipient_country'] else 'international'
            ])
            
            # Add risk-based segmentation
            # risk_level = 'high_risk' if row['risk_score'] > 70 else \
            #             'low_risk' if row['risk_score'] < 30 else 'medium_risk'
            # base_group.append(risk_level)
            
        else:
            # Company peer grouping
            base_group.extend([
                str(row['naics_code']) if pd.notna(row['naics_code']) else 'unknown_naics',
                str(row['amount_segment']),
                # str(row['frequency_segment']),
                # 'domestic' if row['country'] == row['recipient_country'] else 'international'
            ])
            
            # # Add company-specific segments
            # revenue_level = 'high_revenue' if row['annual_revenue'] > 1e7 else \
            #               'low_revenue' if row['annual_revenue'] < 1e6 else 'medium_revenue'
            # base_group.append(revenue_level)
        
        return "_".join(base_group)

    def _create_peer_deviation_features(self, df):
        """Create features based on peer group behavior deviation"""
        # First create sophisticated peer groups
        df = self._create_peer_groups(df)
        group_stats = self._validate_peer_groups(df)
        
        # Calculate peer group statistics
        for group in df['peer_group'].unique():
            group_data = df[df['peer_group'] == group]
            
            # Calculate amount deviation from peer group mean
            group_mean = group_data['amount'].mean()
            group_std = group_data['amount'].std()
            
            mask = df['peer_group'] == group
            df.loc[mask, 'amount_peer_dev'] = (
                (df.loc[mask, 'amount'] - group_mean) / group_std
            )
        
        return df, group_stats

    def _validate_peer_groups(self, df, min_group_size=30, confidence_level=0.95):
        """Validate peer groups using statistical methods and visualizations"""
        
        group_stats = {}
        # print(df['peer_group'].unique())
        # Statistical validation
        for group in df['peer_group'].unique():
            group_data = df[df['peer_group'] == group]
            n_customers = group_data['customer_id'].nunique()
            
            if n_customers >= min_group_size:
                # Basic statistics
                amount_stats = group_data['amount'].describe()
                
                # Normality test
                _, normality_pval = stats.normaltest(group_data['amount'])
                
                # Calculate confidence intervals
                ci = stats.t.interval(
                    confidence_level,
                    len(group_data) - 1,
                    loc=group_data['amount'].mean(),
                    scale=stats.sem(group_data['amount'])
                )
                
                group_stats[group] = {
                    'n_customers': n_customers,
                    'n_transactions': len(group_data),
                    'mean_amount': amount_stats['mean'],
                    'std_amount': amount_stats['std'],
                    'normality_pval': normality_pval,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1]
                }
        # print(group_stats)
        # Visualizations
        self._plot_peer_group_distributions(df, group_stats)
        
        return group_stats
    
    def _plot_peer_group_distributions(self, df, group_stats):
        """Create visualizations for peer group distributions"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Group sizes
        plt.subplot(2, 2, 1)
        sizes = pd.Series({k: v['n_customers'] for k, v in group_stats.items()})
        sizes.plot(kind='bar')
        plt.title('Peer Group Sizes (Number of Customers)')
        plt.xticks(rotation=45)
        
        # Plot 2: Transaction amount distributions
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='peer_group', y='amount')
        plt.title('Transaction Amount Distributions by Peer Group')
        plt.xticks(rotation=45)
        
        # Plot 3: Transaction frequency heatmap
        plt.subplot(2, 2, 3)
        # Handle infinite values before creating frequency bins
        tx_freq = df['tx_frequency'].replace([np.inf, -np.inf], np.nan).fillna(df['tx_frequency'].mean())
        freq_matrix = pd.crosstab(
            df['peer_group'],
            pd.cut(tx_freq, bins=5)
        )
        sns.heatmap(freq_matrix, cmap='YlOrRd')
        plt.title('Transaction Frequency Distribution by Peer Group')
        
        # Plot 4: Risk score distributions
        plt.subplot(2, 2, 4)
        sns.violinplot(data=df, x='peer_group', y='risk_score')
        plt.title('Risk Score Distributions by Peer Group')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/peer_group_analysis.png')
        plt.close()
        
        # Additional statistical summary
        summary_df = pd.DataFrame(group_stats).T
        summary_df.to_csv('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/peer_group_statistics.csv')
        
        return summary_df
