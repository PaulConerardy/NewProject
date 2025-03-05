#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Système d'alerte pour le système Anti-Blanchiment d'Argent (AML).
Fournit des fonctionnalités pour générer et gérer les alertes basées sur les détections.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class Alert:
    """
    Classe représentant une alerte AML pour une entité spécifique.
    Encapsule toutes les détections et calcule un score de risque global.
    """
    
    def __init__(self, entity_id, entity_data=None, transactions=None):
        """
        Initialiser une nouvelle alerte pour une entité.
        
        Paramètres:
        -----------
        entity_id : str
            Identifiant unique de l'entité
        entity_data : dict, optional
            Données de l'entité (type, nom, pays, etc.)
        transactions : DataFrame, optional
            Transactions associées à l'entité
        """
        self.entity_id = entity_id
        self.entity_data = entity_data if entity_data is not None else {}
        self.transactions = transactions
        self.timestamp = datetime.now()
        self.alert_id = f"AML-{self.timestamp.strftime('%Y%m%d')}-{hash(entity_id) % 10000:04d}"
        
        # Initialiser les détections
        self.detections = {
            # Détections de réseau
            'is_bridge_entity': False,
            'network_metrics': {},
            
            # Détections de schémas
            'is_structuring': False,
            'is_smurfing': False,
            'is_rapid_movement': False,
            'is_money_mule': False,
            'structured_deposits': False,
            'underground_banking': False,
            
            # Métriques détaillées
            'money_mule_metrics': {},
            'structured_deposits_metrics': {},
            'underground_banking_metrics': {}
        }
        
        # Score de risque global
        self.risk_score = 0.0
        self.risk_factors = []
        self.priority = "Faible"
    
    def add_network_detection(self, is_bridge_entity=False, network_metrics=None):
        """
        Ajouter des détections basées sur l'analyse de réseau.
        
        Paramètres:
        -----------
        is_bridge_entity : bool
            Si l'entité est une entité pont
        network_metrics : dict
            Métriques de réseau pour l'entité
        """
        self.detections['is_bridge_entity'] = is_bridge_entity
        
        if network_metrics is not None:
            self.detections['network_metrics'] = network_metrics
            
            # Ajouter aux facteurs de risque si c'est une entité pont
            if is_bridge_entity:
                self.risk_factors.append("Entité pont")
                self.risk_score += 20  # Ajouter 20 points pour une entité pont
            
            # Ajouter des points pour les métriques de réseau élevées
            if 'betweenness' in network_metrics and network_metrics['betweenness'] > 0.1:
                self.risk_factors.append("Forte centralité d'intermédiarité")
                self.risk_score += 5 * network_metrics['betweenness'] / 0.1
                
            if 'pagerank' in network_metrics and network_metrics['pagerank'] > 0.05:
                self.risk_factors.append("PageRank élevé")
                self.risk_score += 5 * network_metrics['pagerank'] / 0.05
    
    def add_pattern_detection(self, is_structuring=False, is_smurfing=False, 
                             is_rapid_movement=False, money_mule_data=None,
                             structured_deposits_data=None, underground_banking_data=None):
        """
        Ajouter des détections basées sur les schémas.
        
        Paramètres:
        -----------
        is_structuring : bool
            Si l'entité est impliquée dans la structuration
        is_smurfing : bool
            Si l'entité est impliquée dans le schtroumpfage
        is_rapid_movement : bool
            Si l'entité est impliquée dans des mouvements rapides de fonds
        money_mule_data : dict
            Données de détection de mule financière
        structured_deposits_data : dict
            Données de détection de dépôts structurés
        underground_banking_data : dict
            Données de détection de système bancaire clandestin
        """
        # Mettre à jour les détections
        self.detections['is_structuring'] = is_structuring
        self.detections['is_smurfing'] = is_smurfing
        self.detections['is_rapid_movement'] = is_rapid_movement
        
        # Ajouter aux facteurs de risque et au score
        if is_structuring:
            self.risk_factors.append("Structuration")
            self.risk_score += 15
            
        if is_smurfing:
            self.risk_factors.append("Schtroumpfage")
            self.risk_score += 15
            
        if is_rapid_movement:
            self.risk_factors.append("Mouvement rapide de fonds")
            self.risk_score += 25
        
        # Traiter les données de mule financière
        if money_mule_data is not None and len(money_mule_data) > 0:
            self.detections['is_money_mule'] = True
            self.detections['money_mule_metrics'] = money_mule_data
            
            self.risk_factors.append("Potentielle mule financière")
            
            # Calculer un score basé sur les caractéristiques des mules (max 30 points)
            mule_score = 0
            if 'withdrawal_ratio' in money_mule_data:
                mule_score += money_mule_data['withdrawal_ratio'] * 15
                
            if 'unique_depositors' in money_mule_data and money_mule_data['unique_depositors'] > 3:
                mule_score += 10
                
            if 'time_diff_days' in money_mule_data and money_mule_data['time_diff_days'] < 1:
                mule_score += 5
                
            self.risk_score += mule_score
        
        # Traiter les données de dépôts structurés
        if structured_deposits_data is not None and len(structured_deposits_data) > 0:
            self.detections['structured_deposits'] = True
            self.detections['structured_deposits_metrics'] = structured_deposits_data
            
            self.risk_factors.append("Dépôts structurés")
            self.risk_score += 20
        
        # Traiter les données de système bancaire clandestin
        if underground_banking_data is not None and len(underground_banking_data) > 0:
            self.detections['underground_banking'] = True
            self.detections['underground_banking_metrics'] = underground_banking_data
            
            self.risk_factors.append("Potentiel système bancaire clandestin")
            
            # Utiliser le score de risque calculé dans la détection (max 35 points)
            if 'risk_score' in underground_banking_data:
                self.risk_score += underground_banking_data['risk_score'] * 35
    
    def calculate_priority(self, threshold_medium=50, threshold_high=75):
        """
        Calculer la priorité de l'alerte en fonction du score de risque.
        
        Paramètres:
        -----------
        threshold_medium : float
            Seuil pour la priorité moyenne
        threshold_high : float
            Seuil pour la priorité élevée
        """
        if self.risk_score >= threshold_high:
            self.priority = "Élevée"
        elif self.risk_score >= threshold_medium:
            self.priority = "Moyenne"
        else:
            self.priority = "Faible"
    
    def should_alert(self, threshold=50):
        """
        Déterminer si une alerte doit être générée en fonction du score de risque.
        
        Paramètres:
        -----------
        threshold : float
            Seuil de score de risque pour générer une alerte
            
        Retourne:
        --------
        bool
            True si une alerte doit être générée, False sinon
        """
        return self.risk_score >= threshold
    
    def to_dict(self):
        """
        Convertir l'alerte en dictionnaire pour l'exportation.
        
        Retourne:
        --------
        dict
            Dictionnaire représentant l'alerte
        """
        return {
            'alert_id': self.alert_id,
            'entity_id': self.entity_id,
            'timestamp': self.timestamp,
            'risk_score': self.risk_score,
            'priority': self.priority,
            'risk_factors': '; '.join(self.risk_factors),
            'is_bridge_entity': self.detections['is_bridge_entity'],
            'is_structuring': self.detections['is_structuring'],
            'is_smurfing': self.detections['is_smurfing'],
            'is_rapid_movement': self.detections['is_rapid_movement'],
            'is_money_mule': self.detections['is_money_mule'],
            'structured_deposits': self.detections['structured_deposits'],
            'underground_banking': self.detections['underground_banking']
        }


class AlertSystem:
    """
    Système de gestion des alertes AML.
    Génère et gère les alertes basées sur les résultats de détection.
    """
    
    def __init__(self, alert_threshold=50):
        """
        Initialiser le système d'alerte.
        
        Paramètres:
        -----------
        alert_threshold : float
            Seuil de score de risque pour générer une alerte
        """
        self.alerts = []
        self.alert_threshold = alert_threshold
    
    def generate_alerts(self, entity_data, transaction_data, network_results, pattern_results):
        """
        Générer des alertes basées sur les résultats de détection.
        
        Paramètres:
        -----------
        entity_data : DataFrame
            Données des entités
        transaction_data : DataFrame
            Données des transactions
        network_results : dict
            Résultats de l'analyse de réseau
        pattern_results : dict
            Résultats de la détection de schémas
            
        Retourne:
        --------
        DataFrame
            DataFrame avec les alertes générées
        """
        # Extraire les résultats de l'analyse de réseau
        metrics = network_results.get('metrics', pd.DataFrame())
        bridge_entities = network_results.get('bridge_entities', [])
        
        # Extraire les résultats de la détection de schémas
        structuring_entities = pattern_results.get('structuring_entities', [])
        smurfing_entities = pattern_results.get('smurfing_entities', [])
        rapid_movement_entities = pattern_results.get('rapid_movement_entities', [])
        money_mules = pattern_results.get('money_mules', pd.DataFrame())
        structured_deposits = pattern_results.get('structured_deposits', pd.DataFrame())
        underground_banking = pattern_results.get('underground_banking', pd.DataFrame())
        
        # Créer un ensemble de toutes les entités à évaluer
        all_entities = set(entity_data['entity_id'])
        
        # Générer des alertes pour chaque entité
        for entity_id in all_entities:
            # Filtrer les transactions pour cette entité
            entity_txns = transaction_data[
                (transaction_data['sender_id'] == entity_id) | 
                (transaction_data['receiver_id'] == entity_id)
            ]
            
            if len(entity_txns) == 0:
                continue
            
            # Obtenir les données de l'entité
            entity_info = entity_data[entity_data['entity_id'] == entity_id]
            if not entity_info.empty:
                entity_info = entity_info.iloc[0].to_dict()
            else:
                entity_info = {}
            
            # Créer une nouvelle alerte
            alert = Alert(entity_id, entity_info, entity_txns)
            
            # Ajouter les détections de réseau
            is_bridge = entity_id in bridge_entities
            
            network_metric = {}
            if not metrics.empty:
                entity_metrics = metrics[metrics['entity_id'] == entity_id]
                if not entity_metrics.empty:
                    network_metric = entity_metrics.iloc[0].to_dict()
            
            alert.add_network_detection(is_bridge, network_metric)
            
            # Ajouter les détections de schémas
            is_structuring = entity_id in structuring_entities
            is_smurfing = entity_id in smurfing_entities
            is_rapid_movement = entity_id in rapid_movement_entities
            
            # Obtenir les données de mule financière
            mule_data = {}
            if not money_mules.empty:
                entity_mule = money_mules[money_mules['entity_id'] == entity_id]
                if not entity_mule.empty:
                    mule_data = entity_mule.iloc[0].to_dict()
            
            # Obtenir les données de dépôts structurés
            deposits_data = {}
            if not structured_deposits.empty:
                entity_deposits = structured_deposits[structured_deposits['receiver_id'] == entity_id]
                if not entity_deposits.empty:
                    deposits_data = entity_deposits.iloc[0].to_dict()
            
            # Obtenir les données de système bancaire clandestin
            banking_data = {}
            if not underground_banking.empty:
                entity_banking = underground_banking[underground_banking['entity_id'] == entity_id]
                if not entity_banking.empty:
                    banking_data = entity_banking.iloc[0].to_dict()
            
            alert.add_pattern_detection(
                is_structuring, is_smurfing, is_rapid_movement,
                mule_data, deposits_data, banking_data
            )
            
            # Calculer la priorité
            alert.calculate_priority()
            
            # Ajouter l'alerte si elle dépasse le seuil
            if alert.should_alert(self.alert_threshold):
                self.alerts.append(alert)
        
        # Convertir les alertes en DataFrame
        if self.alerts:
            alerts_df = pd.DataFrame([alert.to_dict() for alert in self.alerts])
            return alerts_df.sort_values('risk_score', ascending=False)
        
        return pd.DataFrame()
    
    def export_alerts(self, output_path):
        """
        Exporter les alertes vers un fichier CSV.
        
        Paramètres:
        -----------
        output_path : str
            Chemin du fichier de sortie
            
        Retourne:
        --------
        bool
            True si l'exportation a réussi, False sinon
        """
        if not self.alerts:
            return False
        
        alerts_df = pd.DataFrame([alert.to_dict() for alert in self.alerts])
        alerts_df.to_csv(output_path, index=False)
        return True
    
    def get_high_priority_alerts(self):
        """
        Obtenir les alertes de priorité élevée.
        
        Retourne:
        --------
        list
            Liste des alertes de priorité élevée
        """
        return [alert for alert in self.alerts if alert.priority == "Élevée"]
    
    def get_medium_priority_alerts(self):
        """
        Obtenir les alertes de priorité moyenne.
        
        Retourne:
        --------
        list
            Liste des alertes de priorité moyenne
        """
        return [alert for alert in self.alerts if alert.priority == "Moyenne"]
    
    def get_low_priority_alerts(self):
        """
        Obtenir les alertes de priorité faible.
        
        Retourne:
        --------
        list
            Liste des alertes de priorité faible
        """
        return [alert for alert in self.alerts if alert.priority == "Faible"]
    
    def get_alerts_by_entity_type(self, entity_type):
        """
        Obtenir les alertes pour un type d'entité spécifique.
        
        Paramètres:
        -----------
        entity_type : str
            Type d'entité à filtrer
            
        Retourne:
        --------
        list
            Liste des alertes pour le type d'entité spécifié
        """
        return [
            alert for alert in self.alerts 
            if 'entity_type' in alert.entity_data and alert.entity_data['entity_type'] == entity_type
        ]
    
    def get_alerts_by_detection_type(self, detection_type):
        """
        Obtenir les alertes pour un type de détection spécifique.
        
        Paramètres:
        -----------
        detection_type : str
            Type de détection à filtrer (ex: 'is_money_mule', 'is_structuring', etc.)
            
        Retourne:
        --------
        list
            Liste des alertes pour le type de détection spécifié
        """
        return [
            alert for alert in self.alerts 
            if detection_type in alert.detections and alert.detections[detection_type]
        ]
    
    def get_alert_summary(self):
        """
        Obtenir un résumé des alertes générées.
        
        Retourne:
        --------
        dict
            Dictionnaire avec le résumé des alertes
        """
        high_priority = len(self.get_high_priority_alerts())
        medium_priority = len(self.get_medium_priority_alerts())
        low_priority = len(self.get_low_priority_alerts())
        
        detection_counts = {
            'bridge_entity': len(self.get_alerts_by_detection_type('is_bridge_entity')),
            'structuring': len(self.get_alerts_by_detection_type('is_structuring')),
            'smurfing': len(self.get_alerts_by_detection_type('is_smurfing')),
            'rapid_movement': len(self.get_alerts_by_detection_type('is_rapid_movement')),
            'money_mule': len(self.get_alerts_by_detection_type('is_money_mule')),
            'structured_deposits': len(self.get_alerts_by_detection_type('structured_deposits')),
            'underground_banking': len(self.get_alerts_by_detection_type('underground_banking'))
        }
        
        return {
            'total_alerts': len(self.alerts),
            'priority': {
                'high': high_priority,
                'medium': medium_priority,
                'low': low_priority
            },
            'detection_types': detection_counts
        }