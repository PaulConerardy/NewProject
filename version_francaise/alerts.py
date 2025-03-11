import pandas as pd
import numpy as np
from rule_based_detection import RuleBasedDetection
from network_analysis import NetworkAnalysis

class Alerts:
    def __init__(self):
        """Initialise le composant d'alertes."""
        self.rule_based = RuleBasedDetection()
        self.network_analysis = NetworkAnalysis()
        self.threshold = 40  # Seuil pour l'activité suspecte
    
    def generate_entity_score(self, entity_data, entities_df, transactions_df, wires_df):
        """Génère un score combiné pour une entité unique."""
        # Obtenir le score basé sur les règles
        rule_score = self.rule_based.calculate_score(entity_data, transactions_df, wires_df)
        
        # Obtenir le score d'analyse de réseau
        network_score = self.network_analysis.calculate_score(
            entity_data['entity_id'], entities_df, transactions_df, wires_df
        )
        
        # Combiner les scores
        total_score = rule_score + network_score
        
        # Créer l'objet d'alerte
        alert = {
            'entity_id': entity_data['entity_id'],
            'rule_score': rule_score,
            'network_score': network_score,
            'total_score': total_score,
            'is_suspicious': total_score >= self.threshold
        }
        
        return alert
    
    def generate_all_alerts(self, entities_df, transactions_df, wires_df):
        """Génère des alertes pour toutes les entités."""
        alerts = []
        
        for _, entity in entities_df.iterrows():
            alert = self.generate_entity_score(entity, entities_df, transactions_df, wires_df)
            alerts.append(alert)
        
        # Convertir en DataFrame pour une analyse plus facile
        alerts_df = pd.DataFrame(alerts)
        
        # Trier par score total en ordre décroissant
        if not alerts_df.empty:
            alerts_df = alerts_df.sort_values(by='total_score', ascending=False)
        
        return alerts_df
    
    def get_suspicious_entities(self, entities_df, transactions_df, wires_df):
        """Retourne uniquement les entités qui atteignent ou dépassent le seuil suspect."""
        all_alerts = self.generate_all_alerts(entities_df, transactions_df, wires_df)
        
        if all_alerts.empty:
            return pd.DataFrame()
        
        suspicious = all_alerts[all_alerts['is_suspicious'] == True]
        return suspicious
    
    def generate_alert_summary(self, entities_df, transactions_df, wires_df):
        """Génère un résumé des alertes."""
        all_alerts = self.generate_all_alerts(entities_df, transactions_df, wires_df)
        
        if all_alerts.empty:
            return {
                'total_entities': 0,
                'suspicious_entities': 0,
                'avg_score': 0,
                'max_score': 0
            }
        
        summary = {
            'total_entities': len(all_alerts),
            'suspicious_entities': len(all_alerts[all_alerts['is_suspicious'] == True]),
            'avg_score': all_alerts['total_score'].mean(),
            'max_score': all_alerts['total_score'].max()
        }
        
        return summary