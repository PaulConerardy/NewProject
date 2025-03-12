import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RuleBasedDetection:
    def __init__(self):
        """Initialise le composant de détection basée sur les règles d'affaires."""
        self.rules = {
            # Indicateurs d'entreprises de services monétaires (ESM)
            "large_wire_transfers_followed_by_outgoing": 10,  # Reçoit soudainement d'importants télévirements suivis de transferts sortants
            "sanctioned_countries_transactions": 10,          # Fait affaire avec des pays sanctionnés
            "split_cash_deposits_same_day": 10,               # Dépôts en espèces fractionnés le même jour
            
            # Indicateurs de banque clandestine
            "suspected_money_mules": 8,                       # Utilisation de mules d'argent suspectées
            "frequent_email_wire_transfers": 8,               # Utilisation fréquente de transferts par courriel et virements internationaux
            "mixed_funds_between_accounts": 8,                # Mélange de fonds entre comptes personnels et d'entreprise
            "high_volume_deposits": 8,                        # Volume élevé de dépôts
            "structured_deposits_below_threshold": 8,         # Dépôts structurés sous le seuil de 10 000$
            "quick_withdrawals_after_deposits": 8,            # Retraits rapides après dépôts
            "foreign_exchange_wires": 8,                      # Virements de sociétés de change
            "inconsistent_activity": 8                        # Activité incompatible avec le profil
        }
        self.max_score = 30  # Score maximum du composant basé sur les règles
        self.suspicious_countries = ['Iran', 'Emirats Arabes Unis', 'Koweit', 'Hong Kong', 'Chine']
        self.threshold_amount = 10000  # Seuil de déclaration standard
        self.margin = 1000  # Marge en dessous du seuil à considérer comme suspecte
    
    def detect_large_wire_transfers_followed_by_outgoing(self, entity_id, transactions_df, wires_df):
        """
        Détecte les entités qui reçoivent soudainement d'importants télévirements ou dépôts en espèces
        suivis d'un nombre accru de télévirements, de chèques destinés à plusieurs tiers.
        """
        score = 0
        
        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements entrants pour cette entité
            incoming_wires = wires_df[(wires_df['party_key'] == entity_id) & (wires_df['sign'] == '+')]
            
            if len(incoming_wires) > 0:
                # Identifier les gros virements (plus de 5000$)
                large_wires = incoming_wires[incoming_wires['amount'] > 5000]
                
                if len(large_wires) > 0:
                    score += 3
                    
                    # Vérifier les transactions sortantes après les dates des gros virements
                    if transactions_df is not None and not transactions_df.empty:
                        # Convertir les dates en datetime
                        large_wire_dates = pd.to_datetime(large_wires['wire_date'], format='%d%b%Y', errors='coerce')
                        
                        if not large_wire_dates.empty:
                            min_date = large_wire_dates.min()
                            
                            # Filtrer les transactions sortantes après cette date
                            transactions_df['trx_date'] = pd.to_datetime(transactions_df['trx_date'], format='%d%b%Y', errors='coerce')
                            outgoing_txns = transactions_df[
                                (transactions_df['party_key'] == entity_id) & 
                                (transactions_df['sign'] == '-') &
                                (transactions_df['trx_date'] > min_date)
                            ]
                            
                            # Vérifier si les transactions sortantes ont augmenté après les gros virements
                            if len(outgoing_txns) > 3:
                                score += 3
                                
                                # Vérifier si les transactions sont destinées à plusieurs tiers
                                # Note: Dans les données synthétiques, nous n'avons pas d'information directe sur les destinataires
                                # Nous utilisons donc le type de transaction comme proxy
                                unique_types = outgoing_txns['transaction_type_desc'].nunique()
                                if unique_types > 1:
                                    score += 4
        
        # Vérifier également les dépôts en espèces importants
        if transactions_df is not None and not transactions_df.empty:
            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df['sign'] == '+')
            ]
            
            if len(cash_deposits) > 0:
                large_deposits = cash_deposits[cash_deposits['amount'] > 5000]
                if len(large_deposits) > 0:
                    score += 2
                    
                    # Vérifier les transactions sortantes après les dates des gros dépôts
                    large_deposit_dates = pd.to_datetime(large_deposits['trx_date'], format='%d%b%Y', errors='coerce')
                    
                    if not large_deposit_dates.empty:
                        min_date = large_deposit_dates.min()
                        
                        outgoing_txns = transactions_df[
                            (transactions_df['party_key'] == entity_id) & 
                            (transactions_df['sign'] == '-') &
                            (transactions_df['trx_date'] > min_date)
                        ]
                        
                        if len(outgoing_txns) > 3:
                            score += 2
        
        return min(score, self.rules["large_wire_transfers_followed_by_outgoing"])
    
    def detect_sanctioned_countries_transactions(self, entity_id, transactions_df, wires_df):
        """
        Détecte les transactions avec des pays sanctionnés ou à haut risque:
        Iran, Emirats Arabes Unis, Koweit, Hong Kong, Chine.
        """
        score = 0
        
        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements pour cette entité
            entity_wires = wires_df[wires_df['party_key'] == entity_id]
            
            if len(entity_wires) > 0:
                # Vérifier les pays d'origine et de destination
                suspicious_origin = entity_wires[entity_wires['originator_country'].isin(['CN', 'HK', 'AE', 'IR', 'KW'])]
                suspicious_dest = entity_wires[entity_wires['beneficiary_country'].isin(['CN', 'HK', 'AE', 'IR', 'KW'])]
                
                # Attribuer un score en fonction du nombre de transactions avec des pays suspects
                suspicious_count = len(suspicious_origin) + len(suspicious_dest)
                
                if suspicious_count > 5:
                    score += 10
                elif suspicious_count > 2:
                    score += 7
                elif suspicious_count > 0:
                    score += 4
        
        return min(score, self.rules["sanctioned_countries_transactions"])
    
    def detect_split_cash_deposits_same_day(self, entity_id, transactions_df, wires_df):
        """
        Détecte les dépôts en espèces importants et fractionnés dans le même compte 
        à plusieurs emplacements le même jour.
        """
        score = 0
        
        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts en espèces pour cette entité
            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df['sign'] == '+')
            ]
            
            if len(cash_deposits) > 0:
                # Convertir les dates en datetime
                cash_deposits['trx_date'] = pd.to_datetime(cash_deposits['trx_date'], format='%d%b%Y', errors='coerce')
                
                # Regrouper par date et compter les emplacements (branches) et montant total
                daily_deposits = cash_deposits.groupby(cash_deposits['trx_date'].dt.date).agg({
                    'branch': 'nunique',
                    'amount': 'sum'
                }).reset_index()
                
                # Vérifier les jours avec plusieurs emplacements et des montants totaux importants
                suspicious_days = daily_deposits[
                    (daily_deposits['branch'] > 1) & 
                    (daily_deposits['amount'] > 5000)
                ]
                
                if len(suspicious_days) > 0:
                    score += min(len(suspicious_days) * 3, 10)
        
        return min(score, self.rules["split_cash_deposits_same_day"])
    
    def detect_suspected_money_mules(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte les mules d'argent suspectées en fonction des modèles de transaction.
        Indicateurs: comptes recevant un grand volume de dépôts de multiples tiers,
        ainsi que des virements internationaux qui ne correspondent pas au profil.
        """
        score = 0
        
        # Vérifier le type d'entité
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        # Vérifier les indicateurs d'activité de mule d'argent
        if transactions_df is not None and not transactions_df.empty:
            # Transactions entrantes
            incoming_txns = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ]
            
            if len(incoming_txns) > 0:
                # Compter les types de transactions entrantes
                deposit_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Depot Especes'])
                email_transfer_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Transfert Internet'])
                
                # Vérifier si le volume est élevé pour un particulier
                if entity_type == 'Particulier' and (deposit_count > 5 or email_transfer_count > 5):
                    score += 3
                
                # Vérifier les transactions sortantes rapides
                outgoing_txns = transactions_df[
                    (transactions_df['party_key'] == entity_id) & 
                    (transactions_df['sign'] == '-')
                ]
                
                if len(outgoing_txns) > 0 and len(incoming_txns) > 0:
                    # Convertir les dates en datetime
                    incoming_txns['trx_date'] = pd.to_datetime(incoming_txns['trx_date'], format='%d%b%Y', errors='coerce')
                    outgoing_txns['trx_date'] = pd.to_datetime(outgoing_txns['trx_date'], format='%d%b%Y', errors='coerce')
                    
                    # Calculer le délai moyen entre dépôts et retraits
                    if not incoming_txns['trx_date'].empty and not outgoing_txns['trx_date'].empty:
                        avg_deposit_date = incoming_txns['trx_date'].mean()
                        avg_withdrawal_date = outgoing_txns['trx_date'].mean()
                        
                        if avg_withdrawal_date > avg_deposit_date:
                            time_diff = (avg_withdrawal_date - avg_deposit_date).days
                            if time_diff <= 3:  # Retraits rapides dans les 3 jours
                                score += 5
        
        # Vérifier les virements internationaux
        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id]
            
            if len(entity_wires) > 0:
                # Pour un particulier, plusieurs virements internationaux peuvent être suspects
                if entity_type == 'Particulier' and len(entity_wires) > 2:
                    score += 3
        
        return min(score, self.rules["suspected_money_mules"])
    
    def detect_frequent_email_wire_transfers(self, entity_id, transactions_df, wires_df):
        """
        Détecte l'utilisation fréquente de transferts par courriel et de virements internationaux.
        """
        score = 0
        
        # Compter les transferts par courriel
        email_count = 0
        if transactions_df is not None and not transactions_df.empty:
            email_transfers = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Transfert Internet')
            ]
            email_count = len(email_transfers)
        
        # Compter les virements internationaux
        wire_count = 0
        if wires_df is not None and not wires_df.empty:
            wires = wires_df[wires_df['party_key'] == entity_id]
            wire_count = len(wires)
        
        # Score basé sur la fréquence combinée
        combined_count = email_count + wire_count
        
        if combined_count > 15:
            score += 8
        elif combined_count > 10:
            score += 6
        elif combined_count > 5:
            score += 3
        elif combined_count > 2:
            score += 1
        
        return min(score, self.rules["frequent_email_wire_transfers"])
    
    def detect_mixed_funds_between_accounts(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte le mélange de fonds entre divers comptes personnels et d'entreprise.
        """
        score = 0
        
        # Vérifier le type d'entité
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        if transactions_df is not None and not transactions_df.empty:
            # Analyser les transactions pour détecter le mélange de fonds
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id]
            
            if len(entity_txns) > 0:
                # Vérifier si l'entité a des transactions avec différents types de comptes
                if 'account_type_desc' in entity_txns.columns:
                    account_types = entity_txns['account_type_desc'].unique()
                    
                    # Si l'entité a des transactions avec plus d'un type de compte, c'est suspect
                    if len(account_types) > 1:
                        score += 4
                        
                        # Si un particulier a des transactions avec des comptes d'entreprise, c'est plus suspect
                        if entity_type == 'Particulier' and 'Entreprise' in account_types:
                            score += 4
        
        return min(score, self.rules["mixed_funds_between_accounts"])
    
    def get_rule_details(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Retourne les détails des règles déclenchées pour une entité spécifique.
        Utile pour expliquer pourquoi une entité a reçu un score élevé.
        """
        rule_details = {}
        
        # Calculer le score pour chaque règle
        rule_details["large_wire_transfers_followed_by_outgoing"] = self.detect_large_wire_transfers_followed_by_outgoing(entity_id, transactions_df, wires_df)
        rule_details["sanctioned_countries_transactions"] = self.detect_sanctioned_countries_transactions(entity_id, transactions_df, wires_df)
        rule_details["split_cash_deposits_same_day"] = self.detect_split_cash_deposits_same_day(entity_id, transactions_df, wires_df)
        rule_details["suspected_money_mules"] = self.detect_suspected_money_mules(entity_id, transactions_df, wires_df, entity_df)
        rule_details["frequent_email_wire_transfers"] = self.detect_frequent_email_wire_transfers(entity_id, transactions_df, wires_df)
        rule_details["mixed_funds_between_accounts"] = self.detect_mixed_funds_between_accounts(entity_id, transactions_df, wires_df, entity_df)
        rule_details["high_volume_deposits"] = self.detect_high_volume_deposits(entity_id, transactions_df, wires_df, entity_df)
        rule_details["structured_deposits_below_threshold"] = self.detect_structured_deposits_below_threshold(entity_id, transactions_df, wires_df)
        rule_details["quick_withdrawals_after_deposits"] = self.detect_quick_withdrawals_after_deposits(entity_id, transactions_df, wires_df)
        rule_details["foreign_exchange_wires"] = self.detect_foreign_exchange_wires(entity_id, transactions_df, wires_df)
        rule_details["inconsistent_activity"] = self.detect_inconsistent_activity(entity_id, transactions_df, wires_df, entity_df)
        
        # Filtrer pour ne garder que les règles avec un score > 0
        triggered_rules = {rule: score for rule, score in rule_details.items() if score > 0}
        
        return triggered_rules
    
    def get_rule_descriptions(self):
        """
        Retourne les descriptions des règles pour l'explication des alertes.
        """
        descriptions = {
            "large_wire_transfers_followed_by_outgoing": "Reçoit soudainement d'importants télévirements suivis de transferts sortants",
            "sanctioned_countries_transactions": "Fait affaire avec des pays sanctionnés ou à haut risque",
            "split_cash_deposits_same_day": "Dépôts en espèces fractionnés le même jour à plusieurs emplacements",
            "suspected_money_mules": "Utilisation de mules d'argent suspectées",
            "frequent_email_wire_transfers": "Utilisation fréquente de transferts par courriel et virements internationaux",
            "mixed_funds_between_accounts": "Mélange de fonds entre comptes personnels et d'entreprise",
            "high_volume_deposits": "Volume inhabituellement élevé de dépôts",
            "structured_deposits_below_threshold": "Dépôts structurés sous le seuil de déclaration de 10 000$",
            "quick_withdrawals_after_deposits": "Retraits rapides après dépôts",
            "foreign_exchange_wires": "Virements importants provenant de sociétés de change",
            "inconsistent_activity": "Activité incompatible avec le profil ou le type d'entreprise"
        }
        
        return descriptions

    def detect_high_volume_deposits(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte un volume inhabituellement élevé de dépôts.
        """
        score = 0
        
        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts pour cette entité
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ]
            
            # Vérifier le type d'entité
            entity_type = None
            if entity_df is not None and not entity_df.empty:
                entity_info = entity_df[entity_df['party_key'] == entity_id]
                if not entity_info.empty:
                    entity_type = entity_info['account_type_desc'].iloc[0]
            
            # Définir des seuils différents selon le type d'entité
            threshold = 10 if entity_type == 'Particulier' else 20
            
            # Calculer le score en fonction du nombre de dépôts
            if len(deposits) > threshold * 2:
                score += 8
            elif len(deposits) > threshold:
                score += 4
        
        return min(score, self.rules["high_volume_deposits"])
    
    def detect_structured_deposits_below_threshold(self, entity_id, transactions_df, wires_df):
        """
        Détecte les dépôts structurés juste en dessous du seuil de déclaration de 10 000$.
        """
        score = 0
        
        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts pour cette entité
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ]
            
            if len(deposits) > 0:
                # Compter les dépôts juste en dessous du seuil
                threshold = self.threshold_amount
                margin = self.margin
                
                structured_deposits = deposits[
                    (deposits['amount'] >= threshold - margin) & 
                    (deposits['amount'] < threshold)
                ]
                
                # Calculer le score en fonction du nombre de dépôts structurés
                structured_count = len(structured_deposits)
                
                if structured_count > 3:
                    score += 8
                elif structured_count > 1:
                    score += 4
                elif structured_count > 0:
                    score += 2
        
        return min(score, self.rules["structured_deposits_below_threshold"])
    
    def detect_quick_withdrawals_after_deposits(self, entity_id, transactions_df, wires_df):
        """
        Détecte les retraits rapides après dépôts.
        """
        score = 0
        
        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les transactions pour cette entité
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id]
            
            if len(entity_txns) > 0:
                # Convertir les dates en datetime
                entity_txns['trx_date'] = pd.to_datetime(entity_txns['trx_date'], format='%d%b%Y', errors='coerce')
                
                # Séparer les dépôts et les retraits
                deposits = entity_txns[entity_txns['sign'] == '+'].sort_values('trx_date')
                withdrawals = entity_txns[entity_txns['sign'] == '-'].sort_values('trx_date')
                
                if not deposits.empty and not withdrawals.empty:
                    # Compter les retraits rapides après dépôts
                    quick_withdrawals = 0
                    
                    for _, deposit in deposits.iterrows():
                        deposit_date = deposit['trx_date']
                        deposit_amount = deposit['amount']
                        
                        # Chercher des retraits dans les 3 jours suivant le dépôt
                        subsequent_withdrawals = withdrawals[
                            (withdrawals['trx_date'] > deposit_date) & 
                            (withdrawals['trx_date'] <= deposit_date + pd.Timedelta(days=3))
                        ]
                        
                        if not subsequent_withdrawals.empty:
                            withdrawal_amount = subsequent_withdrawals['amount'].sum()
                            
                            # Si le montant retiré est proche du montant déposé, c'est suspect
                            if withdrawal_amount >= 0.7 * deposit_amount:
                                quick_withdrawals += 1
                    
                    # Calculer le score en fonction du nombre de retraits rapides
                    if quick_withdrawals > 3:
                        score += 8
                    elif quick_withdrawals > 1:
                        score += 4
                    elif quick_withdrawals > 0:
                        score += 2
        
        return min(score, self.rules["quick_withdrawals_after_deposits"])
    
    def detect_foreign_exchange_wires(self, entity_id, transactions_df, wires_df):
        """
        Détecte les virements importants provenant de sociétés de change.
        """
        score = 0
        
        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements pour cette entité
            entity_wires = wires_df[wires_df['party_key'] == entity_id]
            
            if len(entity_wires) > 0:
                # Identifier les virements provenant de sociétés de change
                # Note: Dans les données synthétiques, nous n'avons pas d'information directe sur le type d'entreprise
                # Nous utilisons donc le nom comme proxy (contient "change", "forex", "exchange", etc.)
                forex_wires = entity_wires[
                    entity_wires['originator'].str.contains('change|forex|exchange|money', case=False, na=False)
                ]
                
                # Calculer le score en fonction du nombre et du montant des virements
                forex_count = len(forex_wires)
                
                if forex_count > 0:
                    score += min(forex_count * 2, 6)
                    
                    # Vérifier si certains virements sont importants
                    large_forex_wires = forex_wires[forex_wires['amount'] > 5000]
                    if len(large_forex_wires) > 0:
                        score += min(len(large_forex_wires) * 2, 4)
        
        return min(score, self.rules["foreign_exchange_wires"])
    
    def detect_inconsistent_activity(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte une activité incompatible avec le profil ou le type d'entreprise.
        """
        score = 0
        
        # Vérifier le type d'entité
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        if transactions_df is not None and not transactions_df.empty and entity_type is not None:
            # Filtrer les transactions pour cette entité
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id]
            
            if len(entity_txns) > 0:
                # Analyser les types de transactions
                txn_types = entity_txns['transaction_type_desc'].value_counts()
                
                # Pour les particuliers, certains types de transactions commerciales sont suspects
                if entity_type == 'Particulier':
                    commercial_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Entreprise|Commercial|Business', case=False, na=False)
                    ]
                    
                    if len(commercial_txns) > 0:
                        score += min(len(commercial_txns), 6)
                
                # Pour les entreprises, un volume élevé de transactions personnelles peut être suspect
                elif entity_type == 'Entreprise':
                    personal_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Personnel|Personal', case=False, na=False)
                    ]
                    
                    if len(personal_txns) > 0:
                        score += min(len(personal_txns), 6)
        
        # Vérifier les virements internationaux incohérents avec le profil
        if wires_df is not None and not wires_df.empty and entity_type is not None:
            entity_wires = wires_df[wires_df['party_key'] == entity_id]
            
            if len(entity_wires) > 0:
                # Pour les particuliers, des virements fréquents vers de nombreux pays différents sont suspects
                if entity_type == 'Particulier':
                    unique_countries = pd.concat([
                        entity_wires['originator_country'].dropna(),
                        entity_wires['beneficiary_country'].dropna()
                    ]).nunique()
                    
                    if unique_countries > 5:
                        score += 4
                    elif unique_countries > 3:
                        score += 2
        
        return min(score, self.rules["inconsistent_activity"])
    
    def calculate_score(self, entity_id, transactions_df, wires_df, entity_df):
        """Calcule le score global basé sur les règles pour une entité."""
        score = 0
        
        # Appliquer chaque règle et additionner les scores
        score += self.detect_large_wire_transfers_followed_by_outgoing(entity_id, transactions_df, wires_df)
        score += self.detect_sanctioned_countries_transactions(entity_id, transactions_df, wires_df)
        score += self.detect_split_cash_deposits_same_day(entity_id, transactions_df, wires_df)
        score += self.detect_suspected_money_mules(entity_id, transactions_df, wires_df, entity_df)
        score += self.detect_frequent_email_wire_transfers(entity_id, transactions_df, wires_df)
        score += self.detect_mixed_funds_between_accounts(entity_id, transactions_df, wires_df, entity_df)
        score += self.detect_high_volume_deposits(entity_id, transactions_df, wires_df, entity_df)
        score += self.detect_structured_deposits_below_threshold(entity_id, transactions_df, wires_df)
        score += self.detect_quick_withdrawals_after_deposits(entity_id, transactions_df, wires_df)
        score += self.detect_foreign_exchange_wires(entity_id, transactions_df, wires_df)
        score += self.detect_inconsistent_activity(entity_id, transactions_df, wires_df, entity_df)
        
        # Normaliser au max_score
        normalized_score = min(score, self.max_score)
        return normalized_score