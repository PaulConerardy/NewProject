import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

class AMLDetectionSystem:
    """
    Système de détection de blanchiment d'argent utilisant des techniques d'apprentissage automatique
    et d'analyse de réseau pour identifier les transactions et entités suspectes.
    """
    
    def __init__(self, transaction_data, entity_data):
        """
        Initialise le système de détection avec les données de transaction et d'entité.
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction avec colonnes : sender_id, receiver_id, amount, timestamp, etc.
        entity_data : DataFrame
            Données d'entité avec colonnes : entity_id, entity_type, country, etc.
        """
        self.transactions = transaction_data
        self.entities = entity_data
        self.features = None
        self.anomaly_scores = None
    
    def preprocess_data(self):
        """
        Prétraite les données de transaction pour l'analyse.
        - Agrège les transactions par entité
        - Calcule les caractéristiques de base
        - Normalise les caractéristiques
        """
        # Calculer les caractéristiques par entité
        entity_features = self._calculate_entity_features()
        
        # Normaliser les caractéristiques
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(entity_features)
        
        self.features = pd.DataFrame(
            normalized_features,
            index=entity_features.index,
            columns=entity_features.columns
        )
    
    def detect_anomalies(self, contamination=0.1):
        """
        Détecte les anomalies dans les données de transaction en utilisant Isolation Forest.
        
        Paramètres:
        -----------
        contamination : float
            Proportion attendue d'anomalies dans le jeu de données
        
        Retourne:
        --------
        DataFrame
            Entités marquées comme anomalies avec leurs scores
        """
        if self.features is None:
            self.preprocess_data()
        
        # Appliquer Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.features)
        
        # Calculer les scores d'anomalie (-1 pour anomalies, 1 pour normal)
        anomaly_scores = iso_forest.score_samples(self.features)
        
        # Créer un DataFrame avec les résultats
        results = pd.DataFrame({
            'entity_id': self.features.index,
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_labels == -1
        })
        
        # Fusionner avec les données d'entité pour plus d'informations
        results = results.merge(self.entities, on='entity_id', how='left')
        
        self.anomaly_scores = results
        return results[results['is_anomaly']].sort_values('anomaly_score')
    
    def _calculate_entity_features(self):
        """
        Calcule les caractéristiques au niveau de l'entité pour la détection d'anomalies.
        
        Retourne:
        --------
        DataFrame
            Caractéristiques calculées par entité
        """
        # Calculer les caractéristiques des transactions sortantes
        outgoing = self.transactions.groupby('sender_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).fillna(0)
        
        # Calculer les caractéristiques des transactions entrantes
        incoming = self.transactions.groupby('receiver_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).fillna(0)
        
        # Renommer les colonnes
        outgoing.columns = [
            'out_txn_count', 'out_amount_sum',
            'out_amount_mean', 'out_amount_std',
            'out_total_txns'
        ]
        incoming.columns = [
            'in_txn_count', 'in_amount_sum',
            'in_amount_mean', 'in_amount_std',
            'in_total_txns'
        ]
        
        # Fusionner les caractéristiques entrantes et sortantes
        features = pd.merge(
            outgoing, incoming,
            left_index=True, right_index=True,
            how='outer'
        ).fillna(0)
        
        # Calculer des ratios et caractéristiques supplémentaires
        features['total_amount'] = features['out_amount_sum'] + features['in_amount_sum']
        features['net_flow'] = features['in_amount_sum'] - features['out_amount_sum']
        features['flow_ratio'] = features['out_amount_sum'] / features['in_amount_sum'].replace(0, 1)
        features['avg_txn_size'] = features['total_amount'] / features['out_total_txns'].replace(0, 1)
        
        return features