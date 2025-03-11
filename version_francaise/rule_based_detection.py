import pandas as pd
import numpy as np

class RuleBasedDetection:
    def __init__(self):
        """Initialise le composant de détection basée sur les règles."""
        self.rules = {
            # Indicateurs MSB
            "large_wire_transfers": 5,
            "suspicious_countries": 5,
            "split_cash_deposits": 5,
            
            # Indicateurs de banque clandestine
            "suspected_mules": 5,
            "frequent_email_wire": 5,
            "mixed_funds": 5,
            "high_volume_deposits": 5,
            "structured_cash_deposits": 5,
            "quick_withdrawals": 5,
            "foreign_exchange_wires": 5,
            "inconsistent_activity": 5
        }
        self.max_score = 25  # Score maximum du composant basé sur les règles
    
    def detect_large_wire_transfers(self, entity_data, transaction_data, wire_data):
        """Détecte les entités recevant de gros virements suivis d'une augmentation des transferts sortants."""
        # Logique d'implémentation
        score = 0
        
        # Vérifier les gros virements entrants
        if wire_data is not None and not wire_data.empty:
            incoming_wires = wire_data[wire_data['destination_id'] == entity_data['entity_id']]
            if len(incoming_wires) > 0:
                large_wires = incoming_wires[incoming_wires['amount'] > 10000]
                if len(large_wires) > 0:
                    score += 2
                    
                    # Vérifier les transactions sortantes ultérieures
                    if transaction_data is not None and not transaction_data.empty:
                        # Filtrer les transactions après les dates des gros virements
                        large_wire_dates = large_wires['date'].unique()
                        if len(large_wire_dates) > 0:
                            outgoing_txns = transaction_data[
                                (transaction_data['sender_id'] == entity_data['entity_id']) & 
                                (transaction_data['date'] > min(large_wire_dates))
                            ]
                            
                            # Vérifier si les transactions sortantes ont augmenté après les gros virements
                            if len(outgoing_txns) > 5:
                                score += 3
        
        return min(score, self.rules["large_wire_transfers"])
    
    def detect_suspicious_countries(self, entity_data, transaction_data, wire_data):
        """Détecte les transactions avec des pays sanctionnés ou des juridictions à haut risque."""
        score = 0
        suspicious_countries = ['Iran', 'United Arab Emirates', 'Kuwait', 'Hong Kong', 'China']
        
        if wire_data is not None and not wire_data.empty:
            # Vérifier les virements sortants
            outgoing_wires = wire_data[wire_data['sender_id'] == entity_data['entity_id']]
            if len(outgoing_wires) > 0:
                suspicious_wires = outgoing_wires[outgoing_wires['destination_country'].isin(suspicious_countries)]
                if len(suspicious_wires) > 0:
                    score += len(suspicious_countries) * 1
            
            # Vérifier les virements entrants
            incoming_wires = wire_data[wire_data['destination_id'] == entity_data['entity_id']]
            if len(incoming_wires) > 0:
                suspicious_wires = incoming_wires[incoming_wires['sender_country'].isin(suspicious_countries)]
                if len(suspicious_wires) > 0:
                    score += len(suspicious_countries) * 1
        
        return min(score, self.rules["suspicious_countries"])
    
    def detect_split_cash_deposits(self, entity_data, transaction_data, wire_data):
        """Détecte les dépôts en espèces importants et fractionnés dans le même compte à plusieurs endroits le même jour."""
        score = 0
        
        if transaction_data is not None and not transaction_data.empty:
            # Filtrer les dépôts en espèces pour cette entité
            cash_deposits = transaction_data[
                (transaction_data['destination_id'] == entity_data['entity_id']) & 
                (transaction_data['type'] == 'cash_deposit')
            ]
            
            if len(cash_deposits) > 0:
                # Regrouper par date et compter les emplacements
                daily_deposits = cash_deposits.groupby('date').agg({
                    'location': 'nunique',
                    'amount': 'sum'
                }).reset_index()
                
                # Vérifier les jours avec plusieurs emplacements et des montants totaux importants
                suspicious_days = daily_deposits[
                    (daily_deposits['location'] > 1) & 
                    (daily_deposits['amount'] > 9000)
                ]
                
                if len(suspicious_days) > 0:
                    score += min(len(suspicious_days), 5)
        
        return min(score, self.rules["split_cash_deposits"])
    
    def detect_suspected_mules(self, entity_data, transaction_data, wire_data):
        """Détecte les mules d'argent suspectées en fonction des modèles de transaction."""
        score = 0
        
        # Vérifier les indicateurs d'activité de mule d'argent
        if transaction_data is not None and not transaction_data.empty:
            # Transactions entrantes de sources multiples
            incoming_txns = transaction_data[transaction_data['destination_id'] == entity_data['entity_id']]
            if len(incoming_txns) > 0:
                unique_senders = incoming_txns['sender_id'].nunique()
                if unique_senders > 5:
                    score += 2
                    
                    # Vérifier les transferts sortants rapides
                    outgoing_txns = transaction_data[transaction_data['sender_id'] == entity_data['entity_id']]
                    if len(outgoing_txns) > 0:
                        # Calculer le temps moyen entre la réception et l'envoi
                        if len(incoming_txns) > 0 and len(outgoing_txns) > 0:
                            # Ceci est simplifié - dans une implémentation réelle, vous associeriez des flux spécifiques entrants/sortants
                            avg_hold_time = 3  # Placeholder pour le calcul réel
                            if avg_hold_time < 2:  # Si les fonds sont détenus pendant moins de 2 jours en moyenne
                                score += 3
        
        return min(score, self.rules["suspected_mules"])
    
    def detect_frequent_email_wire(self, entity_data, transaction_data, wire_data):
        """Détecte l'utilisation fréquente de transferts par courriel et de virements internationaux."""
        score = 0
        
        # Compter les transferts par courriel
        email_count = 0
        if transaction_data is not None and not transaction_data.empty:
            email_transfers = transaction_data[
                ((transaction_data['sender_id'] == entity_data['entity_id']) | 
                 (transaction_data['destination_id'] == entity_data['entity_id'])) &
                (transaction_data['type'] == 'email_transfer')
            ]
            email_count = len(email_transfers)
        
        # Compter les virements
        wire_count = 0
        if wire_data is not None and not wire_data.empty:
            wires = wire_data[
                (wire_data['sender_id'] == entity_data['entity_id']) | 
                (wire_data['destination_id'] == entity_data['entity_id'])
            ]
            wire_count = len(wires)
        
        # Score basé sur la fréquence
        if email_count > 10 and wire_count > 5:
            score += 5
        elif email_count > 5 and wire_count > 2:
            score += 3
        elif email_count > 3 or wire_count > 1:
            score += 1
        
        return min(score, self.rules["frequent_email_wire"])
    
    def detect_mixed_funds(self, entity_data, transaction_data, wire_data):
        """Détecte le mélange de fonds provenant de sources multiples suivi de transferts consolidés."""
        score = 0
        
        if transaction_data is not None and not transaction_data.empty:
            # Vérifier les sources entrantes multiples
            incoming_txns = transaction_data[transaction_data['destination_id'] == entity_data['entity_id']]
            if len(incoming_txns) > 0:
                unique_senders = incoming_txns['sender_id'].nunique()
                
                # Si l'entité reçoit de sources multiples
                if unique_senders >= 3:
                    score += 2
                    
                    # Vérifier les transferts sortants consolidés
                    outgoing_txns = transaction_data[transaction_data['sender_id'] == entity_data['entity_id']]
                    if len(outgoing_txns) > 0:
                        # Si moins de destinations sortantes que de sources entrantes, les fonds sont consolidés
                        unique_destinations = outgoing_txns['destination_id'].nunique()
                        if unique_destinations < unique_senders and unique_destinations <= 2:
                            score += 3
        
        return min(score, self.rules["mixed_funds"])
    
    def detect_high_volume_deposits(self, entity_data, transaction_data, wire_data):
        """Détecte un volume inhabituellement élevé de dépôts sur une courte période."""
        score = 0
        
        if transaction_data is not None and not transaction_data.empty:
            # Filtrer les dépôts pour cette entité
            deposits = transaction_data[
                (transaction_data['destination_id'] == entity_data['entity_id']) & 
                (transaction_data['type'].isin(['deposit', 'cash_deposit']))
            ]
            
            if len(deposits) > 0:
                # Regrouper par date et compter les dépôts
                deposits['date'] = pd.to_datetime(deposits['date'])
                daily_deposits = deposits.groupby(deposits['date'].dt.date).size()
                
                # Vérifier les jours avec un volume élevé de dépôts
                high_volume_days = daily_deposits[daily_deposits > 5].count()
                if high_volume_days > 0:
                    score += min(high_volume_days, 5)
        
        return min(score, self.rules["high_volume_deposits"])
    
    def detect_structured_cash_deposits(self, entity_data, transaction_data, wire_data):
        """Détecte les dépôts en espèces structurés (plusieurs dépôts juste en dessous du seuil de déclaration)."""
        score = 0
        threshold = 10000  # Seuil de déclaration standard
        margin = 1000  # Marge en dessous du seuil à considérer comme suspecte
        
        if transaction_data is not None and not transaction_data.empty:
            # Filtrer les dépôts en espèces pour cette entité
            cash_deposits = transaction_data[
                (transaction_data['destination_id'] == entity_data['entity_id']) & 
                (transaction_data['type'] == 'cash_deposit')
            ]
            
            if len(cash_deposits) > 0:
                # Trouver les dépôts juste en dessous du seuil
                suspicious_deposits = cash_deposits[
                    (cash_deposits['amount'] >= threshold - margin) & 
                    (cash_deposits['amount'] < threshold)
                ]
                
                # Score basé sur le nombre de dépôts suspects
                if len(suspicious_deposits) >= 3:
                    score += 5
                elif len(suspicious_deposits) >= 1:
                    score += len(suspicious_deposits)
        
        return min(score, self.rules["structured_cash_deposits"])
    
    def detect_quick_withdrawals(self, entity_data, transaction_data, wire_data):
        """Détecte les retraits rapides après les dépôts (comportement potentiel de mule d'argent)."""
        score = 0
        
        if transaction_data is not None and not transaction_data.empty:
            # Obtenir les dépôts et les retraits
            deposits = transaction_data[
                (transaction_data['destination_id'] == entity_data['entity_id']) & 
                (transaction_data['type'].isin(['deposit', 'cash_deposit', 'wire_transfer']))
            ]
            
            withdrawals = transaction_data[
                (transaction_data['sender_id'] == entity_data['entity_id']) & 
                (transaction_data['type'].isin(['withdrawal', 'cash_withdrawal', 'wire_transfer']))
            ]
            
            if len(deposits) > 0 and len(withdrawals) > 0:
                # Convertir les dates en datetime si elles ne le sont pas déjà
                deposits['date'] = pd.to_datetime(deposits['date'])
                withdrawals['date'] = pd.to_datetime(withdrawals['date'])
                
                # Trouver le dernier dépôt et le premier retrait
                latest_deposit = deposits['date'].max()
                earliest_withdrawal = withdrawals['date'].min()
                
                # Vérifier s'il y a des retraits peu après les dépôts
                if earliest_withdrawal >= latest_deposit:
                    # Calculer la différence de temps en jours
                    time_diff = (earliest_withdrawal - latest_deposit).days
                    
                    # Score basé sur la rapidité avec laquelle les fonds ont été retirés
                    if time_diff <= 1:  # Le même jour ou le lendemain
                        score += 5
                    elif time_diff <= 3:  # Dans les 3 jours
                        score += 3
                    elif time_diff <= 7:  # Dans la semaine
                        score += 1
        
        return min(score, self.rules["quick_withdrawals"])
    
    def detect_foreign_exchange_wires(self, entity_data, transaction_data, wire_data):
        """Détecte les modèles de transactions de change et de virements internationaux."""
        score = 0
        high_risk_countries = ['Iran', 'North Korea', 'Syria', 'United Arab Emirates', 'Kuwait', 'Hong Kong', 'China']
        
        if wire_data is not None and not wire_data.empty:
            # Vérifier les virements internationaux
            entity_wires = wire_data[
                (wire_data['sender_id'] == entity_data['entity_id']) | 
                (wire_data['destination_id'] == entity_data['entity_id'])
            ]
            
            if len(entity_wires) > 0:
                # Vérifier les pays à haut risque
                high_risk_wires = entity_wires[
                    (entity_wires['sender_country'].isin(high_risk_countries)) | 
                    (entity_wires['destination_country'].isin(high_risk_countries))
                ]
                
                if len(high_risk_wires) > 0:
                    score += min(len(high_risk_wires), 3)
                
                # Vérifier la fréquence des virements internationaux
                if len(entity_wires) >= 5:
                    score += 2
        
        return min(score, self.rules["foreign_exchange_wires"])
    
    def detect_inconsistent_activity(self, entity_data, transaction_data, wire_data):
        """Détecte une activité incompatible avec le profil ou le type d'entreprise de l'entité."""
        score = 0
        
        # Cela utiliserait généralement les informations du profil de l'entité
        # Pour cet exemple, nous utiliserons une approche simplifiée
        
        if transaction_data is not None and not transaction_data.empty:
            # Obtenir toutes les transactions pour cette entité
            entity_txns = transaction_data[
                (transaction_data['sender_id'] == entity_data['entity_id']) | 
                (transaction_data['destination_id'] == entity_data['entity_id'])
            ]
            
            if len(entity_txns) > 0:
                # Vérifier les types de transaction inhabituels
                txn_types = entity_txns['type'].unique()
                
                # Pour les particuliers, plusieurs transactions de type commercial pourraient être suspectes
                if entity_data['entity_type'] == 'individual':
                    business_txn_types = ['business_payment', 'payroll', 'invoice_payment']
                    unusual_types = [t for t in txn_types if t in business_txn_types]
                    
                    if len(unusual_types) > 0:
                        score += min(len(unusual_types) * 2, 5)
                
                # Pour les entreprises, un volume élevé de transferts personnels pourrait être suspect
                elif entity_data['entity_type'] == 'business':
                    personal_txn_types = ['personal_transfer', 'gift']
                    unusual_types = [t for t in txn_types if t in personal_txn_types]
                    
                    if len(unusual_types) > 0:
                        score += min(len(unusual_types) * 2, 5)
        
        return min(score, self.rules["inconsistent_activity"])
    
    # D'autres méthodes de règles seraient implémentées de manière similaire
    
    def calculate_score(self, entity_data, transaction_data, wire_data):
        """Calcule le score global basé sur les règles pour une entité."""
        score = 0
        
        # Appliquer chaque règle et additionner les scores
        score += self.detect_large_wire_transfers(entity_data, transaction_data, wire_data)
        score += self.detect_suspicious_countries(entity_data, transaction_data, wire_data)
        score += self.detect_split_cash_deposits(entity_data, transaction_data, wire_data)
        score += self.detect_suspected_mules(entity_data, transaction_data, wire_data)
        score += self.detect_frequent_email_wire(entity_data, transaction_data, wire_data)
        
        # Ajouter les nouvelles méthodes de détection
        score += self.detect_mixed_funds(entity_data, transaction_data, wire_data)
        score += self.detect_high_volume_deposits(entity_data, transaction_data, wire_data)
        score += self.detect_structured_cash_deposits(entity_data, transaction_data, wire_data)
        score += self.detect_quick_withdrawals(entity_data, transaction_data, wire_data)
        score += self.detect_foreign_exchange_wires(entity_data, transaction_data, wire_data)
        score += self.detect_inconsistent_activity(entity_data, transaction_data, wire_data)
        
        # Normaliser au max_score
        normalized_score = min(score, self.max_score)
        return normalized_score