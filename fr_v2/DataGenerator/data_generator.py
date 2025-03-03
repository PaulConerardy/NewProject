import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TransactionDataGenerator:
    def __init__(self, num_entities=100, num_transactions=1000, start_date="2023-01-01", end_date="2023-12-31"):
        self.num_entities = num_entities
        self.num_transactions = num_transactions
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def generate_entity_data(self):
        """Générer des données d'entité synthétiques"""
        entity_types = ['individual', 'business', 'financial_institution']
        countries = ['US', 'UK', 'CA', 'AU', 'SG', 'HK']
        
        entities = []
        for i in range(self.num_entities):
            entity_type = np.random.choice(entity_types, p=[0.7, 0.2, 0.1])
            entities.append({
                'entity_id': f'E{i:04d}',
                'entity_type': entity_type,
                'entity_name': f'Entity_{i}',
                'country': np.random.choice(countries)
            })
        
        return pd.DataFrame(entities)
    
    def generate_transaction_data(self):
        """Générer des données de transaction synthétiques avec des motifs"""
        transaction_types = ['payment', 'transfer', 'deposit', 'withdrawal']
        
        # Créer des clusters de communautés
        communities = self._create_community_clusters()
        
        transactions = []
        for i in range(self.num_transactions):
            # Sélectionner l'expéditeur et le destinataire en fonction des communautés
            if np.random.random() < 0.7:  # 70% des transactions au sein des communautés
                community = np.random.choice(list(communities.keys()))
                sender = np.random.choice(communities[community])
                receiver = np.random.choice(communities[community])
            else:  # 30% des transactions entre communautés
                comm1, comm2 = np.random.choice(list(communities.keys()), size=2, replace=False)
                sender = np.random.choice(communities[comm1])
                receiver = np.random.choice(communities[comm2])
            
            # Générer l'horodatage
            timestamp = self.start_date + (self.end_date - self.start_date) * np.random.random()
            
            # Générer le montant (distribution log-normale)
            amount = np.random.lognormal(mean=7, sigma=1)  # Génère principalement des petits montants avec quelques grands
            
            transactions.append({
                'transaction_id': f'T{i:06d}',
                'sender_id': sender,
                'receiver_id': receiver,
                'amount': amount,
                'timestamp': timestamp,
                'transaction_type': np.random.choice(transaction_types)
            })
        
        df = pd.DataFrame(transactions)
        return df.sort_values('timestamp')
    
    def _create_community_clusters(self):
        """Créer des communautés synthétiques pour la génération de motifs"""
        communities = {}
        entities = [f'E{i:04d}' for i in range(self.num_entities)]
        
        # Créer 5 communautés principales
        num_communities = 5
        entities_per_community = self.num_entities // num_communities
        
        for i in range(num_communities):
            start_idx = i * entities_per_community
            end_idx = start_idx + entities_per_community
            communities[f'C{i}'] = entities[start_idx:end_idx]
        
        return communities