import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TransactionGenerator:
    def __init__(self, n_customers=1000, n_transactions=10000):
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        
        # Define customer profiles
        self.customer_types = ['individual', 'company']
        self.professions = ['employee', 'self-employed', 'unemployed', 'retired', 'student']
        self.naics_codes = {
            '423990': 'Other Miscellaneous Durable Goods Merchant Wholesalers',
            '453998': 'All Other Miscellaneous Store Retailers',
            '522320': 'Financial Transactions Processing',
            '561499': 'All Other Business Support Services',
            '721110': 'Hotels and Motels'
        }
        self.countries = ['US', 'CA', 'GB', 'FR', 'DE', 'CN', 'JP']
        
        # Generate customer profiles
        self.customer_profiles = self._generate_customer_profiles()
    
    def _generate_customer_profiles(self):
        profiles = {}
        for cust_id in range(1, self.n_customers + 1):
            cust_type = np.random.choice(self.customer_types, p=[0.8, 0.2])
            
            profile = {
                'customer_type': cust_type,
                'country': np.random.choice(self.countries, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05])
            }
            
            if cust_type == 'individual':
                profile['profession'] = np.random.choice(self.professions)
                profile['risk_score'] = np.random.uniform(0, 100)
            else:
                profile['naics_code'] = np.random.choice(list(self.naics_codes.keys()))
                profile['annual_revenue'] = np.random.lognormal(mean=12, sigma=2)
                profile['risk_score'] = np.random.uniform(0, 100)
            
            profiles[cust_id] = profile
        return profiles

    def generate_normal_transactions(self):
        # Previous code remains unchanged until transaction generation
        customer_ids = np.random.randint(1, self.n_customers + 1, self.n_transactions)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        timestamps = [start_date + timedelta(
            seconds=np.random.randint(0, 30*24*60*60)
        ) for _ in range(self.n_transactions)]
        
        # Generate base amounts based on customer type
        amounts = []
        transaction_types = []
        recipient_countries = []
        
        for cust_id in customer_ids:
            profile = self.customer_profiles[cust_id]
            
            # Adjust amount based on customer type
            if profile['customer_type'] == 'company':
                amount = np.random.lognormal(mean=8.0, sigma=1.0)
            else:
                amount = np.random.lognormal(mean=4.0, sigma=0.5)
            
            amounts.append(amount)
            
            # Generate transaction type with probabilities based on customer type
            if profile['customer_type'] == 'company':
                trans_type = np.random.choice(
                    ['transfer', 'withdrawal', 'deposit'], 
                    p=[0.7, 0.15, 0.15]
                )
            else:
                trans_type = np.random.choice(
                    ['transfer', 'withdrawal', 'deposit'], 
                    p=[0.4, 0.35, 0.25]
                )
            
            transaction_types.append(trans_type)
            
            # Generate recipient country
            if trans_type == 'transfer':
                recipient_country = np.random.choice(self.countries)
            else:
                recipient_country = profile['country']
            recipient_countries.append(recipient_country)
        
        # Create DataFrame with enhanced features
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'timestamp': timestamps,
            'amount': amounts,
            'transaction_type': transaction_types,
            'recipient_country': recipient_countries
        })
        
        # Add customer profile information
        df['customer_type'] = df['customer_id'].map(lambda x: self.customer_profiles[x]['customer_type'])
        df['country'] = df['customer_id'].map(lambda x: self.customer_profiles[x]['country'])
        df['risk_score'] = df['customer_id'].map(lambda x: self.customer_profiles[x]['risk_score'])
        
        # Add type-specific information
        df['profession'] = df[df['customer_type'] == 'individual']['customer_id'].map(
            lambda x: self.customer_profiles[x].get('profession'))
        df['naics_code'] = df[df['customer_type'] == 'company']['customer_id'].map(
            lambda x: self.customer_profiles[x].get('naics_code'))
        
        return df

    def inject_anomalies(self, df, anomaly_ratio=0.05):
        n_anomalies = int(len(df) * anomaly_ratio)
        df['is_anomaly'] = 0
        
        # Different types of anomalies
        anomaly_types = [
            'high_amount',
            'rapid_succession',
            'unusual_pattern',
            'high_risk_country',
            'structuring'
        ]
        
        for anomaly_type in anomaly_types:
            n_this_type = n_anomalies // len(anomaly_types)
            
            if anomaly_type == 'high_amount':
                # Unusually large transactions
                indices = np.random.choice(df.index, n_this_type)
                df.loc[indices, 'amount'] *= np.random.uniform(10, 20, n_this_type)
                df.loc[indices, 'is_anomaly'] = 1
                
            elif anomaly_type == 'rapid_succession':
                # Multiple transactions in quick succession
                for _ in range(n_this_type):
                    cust_id = np.random.choice(df['customer_id'].unique())
                    base_time = pd.to_datetime(np.random.choice(df['timestamp']))  # Convert to datetime
                    
                    # Add 3-5 transactions within a short time frame
                    for i in range(np.random.randint(3, 6)):
                        new_row = df.loc[df['customer_id'] == cust_id].iloc[0].copy()
                        new_row['timestamp'] = base_time + timedelta(minutes=i*10)
                        new_row['is_anomaly'] = 1
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
            elif anomaly_type == 'unusual_pattern':
                # Transactions that don't match customer profile
                indices = np.random.choice(df[df['customer_type'] == 'individual'].index, n_this_type)
                df.loc[indices, 'amount'] *= 50  # Individual making company-sized transactions
                df.loc[indices, 'is_anomaly'] = 1
                
            elif anomaly_type == 'high_risk_country':
                # Unusual international transfers
                indices = np.random.choice(df[df['transaction_type'] == 'transfer'].index, n_this_type)
                df.loc[indices, 'recipient_country'] = 'CN'  # Example high-risk country
                df.loc[indices, 'is_anomaly'] = 1
                
            elif anomaly_type == 'structuring':
                # Breaking down large transactions into smaller ones
                for _ in range(n_this_type):
                    cust_id = np.random.choice(df['customer_id'].unique())
                    base_time = pd.to_datetime(np.random.choice(df['timestamp']))
                    large_amount = np.random.uniform(50000, 100000)
                    
                    # Split into multiple transactions just under reporting threshold
                    n_splits = int(large_amount / 9000) + 1
                    split_amount = large_amount / n_splits
                    
                    for i in range(n_splits):
                        new_row = df.loc[df['customer_id'] == cust_id].iloc[0].copy()
                        new_row['timestamp'] = base_time + timedelta(days=i)
                        new_row['amount'] = split_amount
                        new_row['is_anomaly'] = 1
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        return df.sort_values('timestamp').reset_index(drop=True)

    def generate_dataset(self, anomaly_ratio=0.05):
        df = self.generate_normal_transactions()
        df = self.inject_anomalies(df, anomaly_ratio)
        return df