import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from tqdm import tqdm

class TransactionGenerator:
    def __init__(self, n_customers=1000, n_transactions=10000, start_date='2022-01-01', end_date='2023-01-01'):
        self.fake = Faker()
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Initialize customer profiles
        self.customers = self._generate_customers()
        
    def _generate_customers(self):
        customers = []
        for _ in range(self.n_customers):
            customer = {
                'customer_id': self.fake.unique.random_number(digits=8),
                'name': self.fake.name(),
                'address': self.fake.address(),
                'business_type': np.random.choice(['Individual', 'MSB', 'Small Business', 'Corporation'], p=[0.7, 0.1, 0.1, 0.1]),
                'risk_score': np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.2, 0.1]),
                'account_age': np.random.randint(1, 3650),  # Days
                'typical_balance': np.random.lognormal(10, 1)
            }
            customers.append(customer)
        return pd.DataFrame(customers)
    
    def _generate_transaction_amount(self, business_type):
        if business_type == 'Individual':
            return np.random.lognormal(6, 1)  # Most transactions under 10k
        elif business_type == 'MSB':
            return np.random.lognormal(8, 1.5)  # Higher amounts, more variance
        else:
            return np.random.lognormal(7, 1.2)  # Medium amounts
    
    def _generate_suspicious_patterns(self):
        # Implement various suspicious patterns
        patterns = [
            self._generate_structuring(),
            self._generate_rapid_movement(),
            self._generate_msb_unusual()
        ]
        return pd.concat(patterns, ignore_index=True)
    
    def _generate_structuring(self, n_transactions=100):
        # Generate structuring pattern (multiple transactions just under reporting threshold)
        transactions = []
        suspicious_customers = self.customers[self.customers['business_type'].isin(['MSB', 'Small Business'])].sample(n=10)
        
        for _, customer in suspicious_customers.iterrows():
            n_splits = np.random.randint(3, 8)
            total_amount = np.random.uniform(15000, 50000)
            split_amounts = np.random.dirichlet(np.ones(n_splits)) * total_amount
            
            for amount in split_amounts:
                transactions.append({
                    'transaction_id': self.fake.unique.random_number(digits=12),
                    'customer_id': customer['customer_id'],
                    'transaction_date': self.fake.date_between(self.start_date, self.end_date),
                    'amount': amount,
                    'transaction_type': np.random.choice(['wire', 'cash']),
                    'is_suspicious': True,
                    'pattern_type': 'structuring'
                })
        
        return pd.DataFrame(transactions)
    
    def _generate_rapid_movement(self, n_transactions=100):
        # Generate rapid movement pattern (funds moving quickly through accounts)
        transactions = []
        suspicious_customers = self.customers[self.customers['business_type'] == 'MSB'].sample(n=5)
        
        for _, customer in suspicious_customers.iterrows():
            base_date = self.fake.date_between(self.start_date, self.end_date)
            amount = np.random.uniform(50000, 200000)
            
            for i in range(np.random.randint(5, 10)):
                transactions.append({
                    'transaction_id': self.fake.unique.random_number(digits=12),
                    'customer_id': customer['customer_id'],
                    'transaction_date': base_date + timedelta(hours=np.random.randint(1, 24)),
                    'amount': amount * (1 + np.random.uniform(-0.1, 0.1)),
                    'transaction_type': 'wire',
                    'is_suspicious': True,
                    'pattern_type': 'rapid_movement'
                })
        
        return pd.DataFrame(transactions)
    
    def _generate_msb_unusual(self, n_transactions=100):
        # Generate unusual MSB activity
        transactions = []
        suspicious_customers = self.customers[self.customers['business_type'] == 'MSB'].sample(n=5)
        
        for _, customer in suspicious_customers.iterrows():
            for _ in range(np.random.randint(10, 20)):
                transactions.append({
                    'transaction_id': self.fake.unique.random_number(digits=12),
                    'customer_id': customer['customer_id'],
                    'transaction_date': self.fake.date_between(self.start_date, self.end_date),
                    'amount': np.random.uniform(100000, 500000),
                    'transaction_type': np.random.choice(['wire', 'cash']),
                    'is_suspicious': True,
                    'pattern_type': 'unusual_msb'
                })
        
        return pd.DataFrame(transactions)
    
    def generate_transactions(self):
        # Generate normal transactions
        normal_transactions = []
        for _ in tqdm(range(self.n_transactions), desc='Generating normal transactions'):
            customer = self.customers.sample(n=1).iloc[0]
            normal_transactions.append({
                'transaction_id': self.fake.unique.random_number(digits=12),
                'customer_id': customer['customer_id'],
                'transaction_date': self.fake.date_between(self.start_date, self.end_date),
                'amount': self._generate_transaction_amount(customer['business_type']),
                'transaction_type': np.random.choice(['wire', 'cash', 'check'], p=[0.4, 0.3, 0.3]),
                'is_suspicious': False,
                'pattern_type': 'normal'
            })
        
        # Combine normal and suspicious transactions
        all_transactions = pd.concat([
            pd.DataFrame(normal_transactions),
            self._generate_suspicious_patterns()
        ], ignore_index=True)
        
        # Sort by date
        all_transactions['transaction_date'] = pd.to_datetime(all_transactions['transaction_date'])
        all_transactions = all_transactions.sort_values('transaction_date')
        
        return all_transactions, self.customers