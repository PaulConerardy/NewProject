#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Schémas de détection pour le système Anti-Blanchiment d'Argent (AML).
Contient les implémentations de diverses méthodes de détection pour identifier
les schémas suspects associés au blanchiment d'argent.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import networkx as nx

class DetectionPatterns:
    """
    Implémente diverses méthodes de détection pour identifier les schémas suspects
    associés au blanchiment d'argent.
    """
    
    @staticmethod
    def detect_structuring(transaction_data, threshold=10000, margin_percent=10):
        """
        Détecter les schémas de structuration (transactions juste en dessous des seuils de déclaration).
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        threshold : float
            Le montant seuil de déclaration
        margin_percent : float
            Pourcentage en dessous du seuil à considérer comme suspect
            
        Retourne:
        --------
        list
            Liste des ID d'entités avec des schémas de structuration suspects
        """
        margin = threshold * (margin_percent / 100)
        lower_bound = threshold - margin
        
        # Trouver les transactions juste en dessous du seuil
        suspicious = transaction_data[
            (transaction_data['amount'] >= lower_bound) & 
            (transaction_data['amount'] < threshold)
        ]
        
        # Grouper par expéditeur pour trouver les entités avec plusieurs transactions similaires
        structuring_entities = suspicious.groupby('sender_id').size()
        structuring_entities = structuring_entities[structuring_entities > 1].index.tolist()
        
        return structuring_entities
    
    @staticmethod
    def detect_smurfing(transaction_data, time_window_days=7, min_transactions=3):
        """
        Détecter les schémas de schtroumpfage (multiples petits dépôts par différents individus sur le même compte).
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        time_window_days : int
            Fenêtre temporelle à considérer pour le regroupement des transactions
        min_transactions : int
            Nombre minimum de transactions à considérer comme suspect
            
        Retourne:
        --------
        list
            Liste des ID d'entités avec des schémas de schtroumpfage suspects
        """
        # Créer une copie pour éviter de modifier l'original
        data = transaction_data.copy()
        
        # Grouper les transactions par destinataire et fenêtre temporelle
        data['time_window'] = data['timestamp'].dt.floor(f'{time_window_days}D')
        
        # Trouver les destinataires avec plusieurs petites transactions de différents expéditeurs
        smurfing = data.groupby(['receiver_id', 'time_window']).agg({
            'sender_id': 'nunique',
            'amount': ['count', 'mean']
        })
        
        smurfing.columns = ['_'.join(col).strip() for col in smurfing.columns.values]
        
        # Filtrer pour les schémas suspects
        suspicious = smurfing[
            (smurfing['sender_id_nunique'] >= min_transactions) & 
            (smurfing['amount_mean'] < smurfing['amount_mean'].quantile(0.25))
        ]
        
        return suspicious.index.get_level_values('receiver_id').unique().tolist()
    
    @staticmethod
    def detect_rapid_movement(transaction_data, entity_data, network_graph, max_hours=48, min_hops=3):
        """
        Détecter les fonds se déplaçant rapidement à travers plusieurs comptes.
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        entity_data : DataFrame
            Données d'entité avec informations sur les entités
        network_graph : NetworkX DiGraph
            Graphe du réseau de transactions
        max_hours : int
            Fenêtre temporelle maximale pour considérer les mouvements comme rapides
        min_hops : int
            Nombre minimum de sauts pour considérer comme suspect
            
        Retourne:
        --------
        list
            Liste des ID d'entités impliquées dans des mouvements rapides de fonds
        """
        # Créer un graphe temporel
        temporal_paths = []
        
        # Trouver les chemins dans le graphe
        for source in network_graph.nodes():
            for target in network_graph.nodes():
                if source != target:
                    paths = nx.all_simple_paths(network_graph, source, target, cutoff=min_hops)
                    temporal_paths.extend(paths)
        
        suspicious_entities = set()
        
        # Analyser chaque chemin pour les mouvements rapides
        for path in temporal_paths:
            if len(path) >= min_hops:
                # Obtenir les transactions le long du chemin
                path_transactions = []
                for i in range(len(path)-1):
                    txns = transaction_data[
                        (transaction_data['sender_id'] == path[i]) &
                        (transaction_data['receiver_id'] == path[i+1])
                    ]
                    if not txns.empty:
                        path_transactions.append(txns.iloc[0])
                
                if len(path_transactions) >= min_hops:
                    # Vérifier le temps total écoulé
                    start_time = min(txn['timestamp'] for txn in path_transactions)
                    end_time = max(txn['timestamp'] for txn in path_transactions)
                    elapsed_hours = (end_time - start_time).total_seconds() / 3600
                    
                    if elapsed_hours <= max_hours:
                        suspicious_entities.update(path)
        
        return list(suspicious_entities)

    @staticmethod
    def detect_money_mules(transaction_data, entity_data, threshold_days=3, min_deposits=5):
        """
        Détecter les potentielles mules financières basées sur les modèles de dépôt et retrait rapide.
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        entity_data : DataFrame
            Données d'entité avec informations sur les entités
        threshold_days : int
            Nombre maximum de jours entre les dépôts et les retraits pour être considéré suspect
        min_deposits : int
            Nombre minimum de dépôts de tiers pour être considéré suspect
            
        Retourne:
        --------
        DataFrame
            DataFrame avec les entités suspectes et leurs métriques
        """
        # Créer une copie pour éviter de modifier l'original
        data = transaction_data.copy()
        
        # S'assurer que le timestamp est en datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Identifier les dépôts (transactions entrantes)
        deposits = data[data['transaction_type'].isin(['deposit', 'transfer'])]
        
        # Identifier les retraits (transactions sortantes)
        withdrawals = data[data['transaction_type'].isin(['withdrawal', 'transfer'])]
        
        # Grouper par destinataire pour trouver les comptes avec plusieurs dépôts de tiers
        deposit_counts = deposits.groupby('receiver_id').size()
        potential_mules = deposit_counts[deposit_counts >= min_deposits].index.tolist()
        
        results = []
        
        for entity_id in potential_mules:
            # Obtenir les dépôts pour cette entité
            entity_deposits = deposits[deposits['receiver_id'] == entity_id]
            
            # Obtenir les retraits pour cette entité
            entity_withdrawals = withdrawals[withdrawals['sender_id'] == entity_id]
            
            if len(entity_withdrawals) == 0:
                continue
            
            # Calculer le délai moyen entre le dernier dépôt et le premier retrait
            last_deposit = entity_deposits['timestamp'].max()
            first_withdrawal = entity_withdrawals['timestamp'].min()
            
            if pd.isna(last_deposit) or pd.isna(first_withdrawal):
                continue
            
            time_diff = (first_withdrawal - last_deposit).total_seconds() / (24*3600)  # en jours
            
            # Si le retrait est rapide après les dépôts, c'est suspect
            if time_diff <= threshold_days:
                # Calculer le nombre de déposants uniques
                unique_depositors = entity_deposits['sender_id'].nunique()
                
                # Calculer le montant total des dépôts et retraits
                total_deposits = entity_deposits['amount'].sum()
                total_withdrawals = entity_withdrawals['amount'].sum()
                
                # Calculer le ratio de retrait (proche de 1 signifie que presque tout l'argent est retiré)
                withdrawal_ratio = total_withdrawals / total_deposits if total_deposits > 0 else 0
                
                results.append({
                    'entity_id': entity_id,
                    'unique_depositors': unique_depositors,
                    'total_deposits': total_deposits,
                    'total_withdrawals': total_withdrawals,
                    'withdrawal_ratio': withdrawal_ratio,
                    'time_diff_days': time_diff
                })
        
        return pd.DataFrame(results) if results else pd.DataFrame()

    @staticmethod
    def detect_structured_deposits(transaction_data, threshold=10000, window_days=30):
        """
        Détecter les dépôts structurés (multiples dépôts sous le seuil de déclaration).
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        threshold : float
            Le montant seuil de déclaration
        window_days : int
            Fenêtre temporelle pour regrouper les transactions
            
        Retourne:
        --------
        DataFrame
            DataFrame avec les entités suspectes et leurs métriques
        """
        # Créer une copie pour éviter de modifier l'original
        data = transaction_data.copy()
        
        # S'assurer que le timestamp est en datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Filtrer pour ne garder que les dépôts
        deposits = data[data['transaction_type'] == 'deposit']
        
        # Filtrer pour les dépôts sous le seuil
        under_threshold = deposits[deposits['amount'] < threshold]
        
        # Ajouter une colonne pour la fenêtre temporelle
        under_threshold['time_window'] = under_threshold['timestamp'].dt.floor(f'{window_days}D')
        
        # Grouper par destinataire et fenêtre temporelle
        grouped = under_threshold.groupby(['receiver_id', 'time_window']).agg({
            'amount': ['count', 'sum'],
            'sender_id': 'nunique'
        })
        
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        # Filtrer pour les cas où il y a plusieurs dépôts sous le seuil dans la même fenêtre
        # et où la somme est supérieure au seuil
        suspicious = grouped[
            (grouped['amount_count'] >= 3) & 
            (grouped['amount_sum'] >= threshold) &
            (grouped['sender_id_nunique'] >= 2)  # Au moins 2 expéditeurs différents
        ]
        
        # Réinitialiser l'index pour obtenir un DataFrame plat
        return suspicious.reset_index()

    @staticmethod
    def detect_underground_banking(transaction_data, entity_data, min_flow_balance=0.8, min_intl_ratio=0.3):
        """
        Détecte les potentiels systèmes bancaires clandestins.
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        entity_data : DataFrame
            Données d'entité avec informations sur les entités
        min_flow_balance : float
            Ratio minimum de balance de flux pour être considéré suspect
        min_intl_ratio : float
            Ratio minimum de transactions internationales pour être considéré suspect
        
        Retourne:
        --------
        DataFrame
            Entités suspectes avec leurs métriques
        """
        # Créer une copie pour éviter de modifier l'original
        data = transaction_data.copy()
        
        # S'assurer que le timestamp est en datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calculer les caractéristiques par entité
        entity_features = {}
        
        # 1. Ratio de transactions internationales
        if 'country' in entity_data.columns:
            # Créer un dictionnaire pour mapper les entités à leurs pays
            entity_countries = dict(zip(entity_data['entity_id'], entity_data['country']))
            
            # Ajouter les pays d'origine et de destination
            data['sender_country'] = data['sender_id'].map(entity_countries)
            data['receiver_country'] = data['receiver_id'].map(entity_countries)
            
            # Calculer le ratio de transactions internationales par entité
            for entity_id in entity_data['entity_id']:
                entity_txns = data[
                    (data['sender_id'] == entity_id) | 
                    (data['receiver_id'] == entity_id)
                ]
                
                if len(entity_txns) == 0:
                    continue
                    
                intl_txns = entity_txns[entity_txns['sender_country'] != entity_txns['receiver_country']]
                
                intl_ratio = len(intl_txns) / len(entity_txns) if len(entity_txns) > 0 else 0
                
                if entity_id not in entity_features:
                    entity_features[entity_id] = {}
                
                entity_features[entity_id]['intl_txn_ratio'] = intl_ratio
        
        # 2. Ratio de transactions sous le seuil de déclaration
        threshold = 10000
        for entity_id in entity_data['entity_id']:
            entity_txns = data[
                (data['sender_id'] == entity_id) | 
                (data['receiver_id'] == entity_id)
            ]
            
            if len(entity_txns) == 0:
                continue
                
            under_threshold_txns = entity_txns[entity_txns['amount'] < threshold]
            
            under_threshold_ratio = len(under_threshold_txns) / len(entity_txns)
            
            if entity_id not in entity_features:
                entity_features[entity_id] = {}
            
            entity_features[entity_id]['under_threshold_ratio'] = under_threshold_ratio
        
        # 3. Ratio de flux équilibrés (entrées vs sorties)
        for entity_id in entity_data['entity_id']:
            incoming = data[data['receiver_id'] == entity_id]['amount'].sum()
            outgoing = data[data['sender_id'] == entity_id]['amount'].sum()
            
            # Un ratio proche de 1 indique des flux équilibrés
            flow_balance = min(incoming, outgoing) / max(incoming, outgoing) if max(incoming, outgoing) > 0 else 0
            
            if entity_id not in entity_features:
                entity_features[entity_id] = {}
            
            entity_features[entity_id]['flow_balance_ratio'] = flow_balance
        
        # Filtrer les entités suspectes
        suspicious_entities = []
        
        for entity_id, features in entity_features.items():
            if 'flow_balance_ratio' in features and 'intl_txn_ratio' in features and 'under_threshold_ratio' in features:
                if (features['flow_balance_ratio'] >= min_flow_balance and 
                    features['intl_txn_ratio'] >= min_intl_ratio and 
                    features['under_threshold_ratio'] >= 0.5):  # Au moins 50% des transactions sous le seuil
                    
                    # Calculer un score de risque
                    risk_score = (
                        features['flow_balance_ratio'] * 0.4 +
                        features['intl_txn_ratio'] * 0.3 +
                        features['under_threshold_ratio'] * 0.3
                    )
                    
                    suspicious_entities.append({
                        'entity_id': entity_id,
                        'flow_balance_ratio': features['flow_balance_ratio'],
                        'intl_txn_ratio': features['intl_txn_ratio'],
                        'under_threshold_ratio': features['under_threshold_ratio'],
                        'risk_score': risk_score
                    })
        
        return pd.DataFrame(suspicious_entities) if suspicious_entities else pd.DataFrame()