import pandas as pd
import numpy as np
from data_generator import TransactionDataGenerator
import os
import random
from datetime import datetime, timedelta

class AnomalyTransactionGenerator:
    """
    Génère des données de transaction détaillées pour les entités identifiées comme
    anomalies dans l'analyse des groupes de pairs.
    """
    
    def __init__(self, anomaly_file, output_dir=None, seed=None):
        """
        Initialise le générateur avec les données d'anomalie.
        
        Paramètres:
        -----------
        anomaly_file : str
            Chemin vers le fichier CSV contenant les données d'anomalie
        output_dir : str, optionnel
            Répertoire pour sauvegarder les fichiers de sortie
        seed : int, optionnel
            Graine aléatoire pour la reproductibilité
        """
        self.anomalies = pd.read_csv(anomaly_file)
        self.output_dir = output_dir or '/Users/paulconerardy/Documents/Trae/ESM3/anomaly_data'
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Définir la graine aléatoire si fournie
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_entity_data(self):
        """
        Générer des données d'entité pour les anomalies.
        
        Retourne:
        --------
        DataFrame
            Données d'entité pour les anomalies
        """
        entities = []
        
        for _, anomaly in self.anomalies.iterrows():
            party_key = str(anomaly['party_key'])
            
            # Mapper le type de partie au type d'entité
            if anomaly['party_type'] == 'Individual':
                entity_type = 'individual'
            elif anomaly['party_type'] == 'Business':
                entity_type = 'business'
            else:
                entity_type = 'financial_institution'
            
            # Créer l'enregistrement d'entité
            entity = {
                'entity_id': party_key,
                'entity_type': entity_type,
                'entity_name': f"Entity_{party_key}",
                'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'SG', 'HK'])
            }
            
            entities.append(entity)
        
        return pd.DataFrame(entities)
    
    def generate_transaction_data(self, num_transactions_per_entity=100):
        """
        Générer des données de transaction pour les anomalies basées sur leurs profils.
        
        Paramètres:
        -----------
        num_transactions_per_entity : int
            Nombre de transactions à générer par entité anomale
            
        Retourne:
        --------
        DataFrame
            Données de transaction pour les anomalies
        """
        transactions = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Créer un pool de contreparties (entités non anomales)
        counterparty_pool = [f"E{i:04d}" for i in range(1000)]
        
        # Générer des transactions pour chaque anomalie
        for _, anomaly in self.anomalies.iterrows():
            party_key = str(anomaly['party_key'])
            
            # Déterminer les modèles de transaction basés sur le type d'anomalie
            # Vérifier s'il s'agit d'une entité bancaire souterraine potentielle
            is_underground = False
            if 'flow_balance' in anomaly and anomaly['flow_balance'] < 0.2:
                is_underground = True
            
            # Vérifier s'il s'agit d'une entité potentielle de blanchiment d'argent basé sur le commerce
            is_trade_ml = False
            if 'intl_txn_ratio' in anomaly and anomaly['intl_txn_ratio'] > 5:
                is_trade_ml = True
            
            # Générer des transactions
            for i in range(num_transactions_per_entity):
                # Déterminer la direction de la transaction (entrante/sortante)
                is_outgoing = random.random() < 0.5
                
                # Pour la banque souterraine, créer des flux équilibrés
                if is_underground:
                    # Créer des paires de transactions avec des montants similaires
                    if i % 2 == 0:  # Les transactions paires sont sortantes
                        is_outgoing = True
                    else:  # Les transactions impaires sont entrantes
                        is_outgoing = False
                
                # Pour le blanchiment d'argent basé sur le commerce, créer plus de transactions internationales
                if is_trade_ml and random.random() < 0.7:
                    # 70% de chance de transaction internationale
                    counterparty_country = np.random.choice(['HK', 'SG', 'UK'])
                else:
                    counterparty_country = 'US'  # Pays par défaut
                
                # Sélectionner la contrepartie
                counterparty = np.random.choice(counterparty_pool)
                
                # Générer le montant de la transaction
                # Montants plus élevés pour les anomalies
                amount = np.random.lognormal(mean=8, sigma=1.5)
                
                # Pour la banque souterraine, rendre les transactions appariées similaires
                if is_underground and i % 2 == 1:
                    # Obtenir le montant de la transaction précédente et ajouter une petite variation
                    prev_amount = transactions[-1]['amount']
                    amount = prev_amount * (1 + random.uniform(-0.05, 0.05))
                
                # Générer l'horodatage
                days_range = (end_date - start_date).days
                random_days = random.randint(0, days_range)
                timestamp = start_date + timedelta(days=random_days)
                
                # Déterminer le type de transaction
                if is_outgoing:
                    sender_id = party_key
                    receiver_id = counterparty
                    transaction_type = np.random.choice(['payment', 'transfer'])
                else:
                    sender_id = counterparty
                    receiver_id = party_key
                    transaction_type = np.random.choice(['deposit', 'transfer'])
                
                # Créer l'enregistrement de transaction
                transaction = {
                    'transaction_id': f"T{len(transactions):06d}",
                    'sender_id': sender_id,
                    'receiver_id': receiver_id,
                    'amount': amount,
                    'timestamp': timestamp,
                    'transaction_type': transaction_type,
                    'sender_country': 'US' if sender_id == party_key else counterparty_country,
                    'receiver_country': 'US' if receiver_id == party_key else counterparty_country
                }
                
                transactions.append(transaction)
        
        # Convertir en DataFrame et trier par horodatage
        df = pd.DataFrame(transactions)
        return df.sort_values('timestamp')
    
    def generate_and_save_data(self):
        """
        Générer et sauvegarder les données d'entité et de transaction pour les anomalies.
        
        Retourne:
        --------
        tuple
            (données_entité, données_transaction)
        """
        print("Génération des données d'entité pour les anomalies...")
        entity_data = self.generate_entity_data()
        
        print("Génération des données de transaction pour les anomalies...")
        transaction_data = self.generate_transaction_data()
        
        # Sauvegarder les données dans des fichiers CSV
        entity_file = os.path.join(self.output_dir, 'anomaly_entities.csv')
        transaction_file = os.path.join(self.output_dir, 'anomaly_transactions.csv')
        
        entity_data.to_csv(entity_file, index=False)
        transaction_data.to_csv(transaction_file, index=False)
        
        print(f"Généré {len(entity_data)} entités et {len(transaction_data)} transactions")
        print(f"Données d'entité sauvegardées dans : {entity_file}")
        print(f"Données de transaction sauvegardées dans : {transaction_file}")
        
        return entity_data, transaction_data

def main():
    # Générer des données pour les principales anomalies
    generator = AnomalyTransactionGenerator(
        anomaly_file='/Users/paulconerardy/Documents/Trae/ESM3/top_anomalies.csv',
        seed=42
    )
    
    # Générer et sauvegarder les données
    entity_data, transaction_data = generator.generate_and_save_data()
    
    print("\nExemple de données d'entité :")
    print(entity_data.head())
    
    print("\nExemple de données de transaction :")
    print(transaction_data.head())
    
    print("\nVous pouvez maintenant exécuter l'analyse de réseau sur ces anomalies en utilisant main.py")

if __name__ == "__main__":
    main()