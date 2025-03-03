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