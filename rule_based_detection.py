import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import yaml


class RuleBasedDetection:
    def __init__(self, config_path=None):
        """
        Initialise le composant de détection basé sur les règles.
        
        Args:
            config_path (str, optional): Chemin vers le fichier config.yaml. 
                                       Si None, utiliser les poids de règles par défaut.
        """
        # Configuration par défaut
        self.config = {
            'rules': {},
            'alert_threshold': 40,
            'prior_suspicious_flag_boost': 20,
            'detection_params': {
                'suspicious_countries': ['Iran', 'Emirats Arabes Unis', 'Koweit', 'Hong Kong', 'Chine'],
                'threshold_amount': 10000,
                'margin': 1000,
                'max_score': 100
            }
        }
        
        # Charger la configuration si fournie
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        
        # Définir les paramètres de détection
        self.alert_threshold = self.config['alert_threshold']
        self.prior_suspicious_flag_boost = self.config['prior_suspicious_flag_boost']
        self.suspicious_countries = self.config['detection_params']['suspicious_countries']
        self.threshold_amount = self.config['detection_params']['threshold_amount']
        self.margin = self.config['detection_params']['margin']
        self.max_score = self.config['detection_params']['max_score']
    
    def _get_graduated_score(self, rule_name, count):
        """
        Obtenir le score approprié basé sur le compteur et les seuils de la règle.
        
        Args:
            rule_name (str): Nom de la règle
            count (int): Nombre d'instances détectées
            
        Returns:
            int: Score basé sur les seuils
        """
        if rule_name not in self.config['rules']:
            return 0
            
        rule_config = self.config['rules'][rule_name]
        thresholds = rule_config['thresholds']
        scores = rule_config['scores']
        
        # Trouver le seuil le plus élevé que le compteur dépasse
        for i in range(len(thresholds) - 1, -1, -1):
            if count >= thresholds[i]:
                return scores[i]
        
        return 0
    
    def _convert_dates(self, df, date_column):
        """
        Méthode auxiliaire pour convertir les colonnes de dates en toute sécurité.
        
        Args:
            df (pd.DataFrame): DataFrame contenant la colonne de date
            date_column (str): Nom de la colonne contenant les dates
            
        Returns:
            pd.DataFrame: DataFrame avec les dates converties
        """
        df = df.copy()
        try:
            # Essayer plusieurs formats de date
            df[date_column] = pd.to_datetime(df[date_column], format='%d%b%Y', errors='coerce')
            if df[date_column].isna().all():
                # Si toutes les dates sont NaT, essayer sans format spécifique
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception:
            # Si la conversion échoue, essayer sans format spécifique
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        return df
    
    def detect_large_wire_transfers_followed_by_outgoing(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les entités qui reçoivent soudainement des virements importants ou des dépôts
        en espèces suivis d'une augmentation des transferts sortants.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements entrants pour cette entité
            incoming_wires = wires_df[(wires_df['party_key'] == entity_id) & (wires_df['sign'] == '+')].copy()
            
            if len(incoming_wires) > 0:
                # Identifier les gros virements (plus de 5000$)
                large_wires = incoming_wires[incoming_wires['amount'] > 5000]
                
                if len(large_wires) > 0:
                    count += 1
                    flagged_wires = pd.concat([flagged_wires, large_wires]) # Collect triggered wires
                    
                    # Vérifier les transactions sortantes après les dates des gros virements
                    if transactions_df is not None and not transactions_df.empty:
                        large_wires = self._convert_dates(large_wires, 'wire_date')
                        
                        if not large_wires.empty:
                            min_date = pd.to_datetime(large_wires['wire_date'].min(), format='%d%b%Y', errors='coerce')
                            
                            transactions_df = self._convert_dates(transactions_df, 'trx_date')

                            outgoing_txns = transactions_df[
                                (transactions_df['party_key'] == entity_id) & 
                                (transactions_df['sign'] == '-') & 
                                (transactions_df['trx_date'] > min_date)
                            ].copy()
                            
                            if len(outgoing_txns) > 3:
                                count += 1
                                flagged_transactions = pd.concat([flagged_transactions, outgoing_txns]) # Collect triggered transactions
                                
                                unique_types = outgoing_txns['transaction_type_desc'].nunique()
                                if unique_types > 1:
                                    count += 1
        
        # Vérifier aussi les gros dépôts en espèces
        if transactions_df is not None and not transactions_df.empty:
            transactions_df = self._convert_dates(transactions_df, 'trx_date')

            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Depot Especes') & 
                (transactions_df['sign'] == '+')
            ].copy()
            
            if len(cash_deposits) > 0:
                large_deposits = cash_deposits[cash_deposits['amount'] > 5000]
                if len(large_deposits) > 0:
                    count += 1
                    flagged_transactions = pd.concat([flagged_transactions, large_deposits]) # Collect triggered deposits
                    
                if not large_deposits.empty:
                    min_date = large_deposits['trx_date'].min()
                    
                    outgoing_txns = transactions_df[
                        (transactions_df['party_key'] == entity_id) & 
                        (transactions_df['sign'] == '-') & 
                        (transactions_df['trx_date'] > min_date)
                    ].copy()
                    
                    if len(outgoing_txns) > 3:
                        count += 1
                        flagged_transactions = pd.concat([flagged_transactions, outgoing_txns]) # Collect triggered transactions
        
        return count, flagged_transactions, flagged_wires
    
    def detect_sanctioned_countries_transactions(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les transactions avec des pays sanctionnés ou à haut risque.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            
            if len(entity_wires) > 0:
                suspicious_origin = entity_wires[entity_wires['originator_country'].isin(['CN', 'HK', 'AE', 'IR', 'KW'])]
                suspicious_dest = entity_wires[entity_wires['beneficiary_country'].isin(['CN', 'HK', 'AE', 'IR', 'KW'])]
                
                count = len(suspicious_origin) + len(suspicious_dest)
                
                if not suspicious_origin.empty:
                    flagged_wires = pd.concat([flagged_wires, suspicious_origin])
                if not suspicious_dest.empty:
                    flagged_wires = pd.concat([flagged_wires, suspicious_dest])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_split_cash_deposits_same_day(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les dépôts en espèces fractionnés le même jour.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts en espèces
            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Depot Especes') & 
                (transactions_df['sign'] == '+')
            ].copy()
            
            if len(cash_deposits) > 0:
                # Convertir les dates
                cash_deposits = self._convert_dates(cash_deposits, 'trx_date')
                
                # Supprimer les lignes où la conversion de date a échoué
                cash_deposits = cash_deposits.dropna(subset=['trx_date'])
                
                if not cash_deposits.empty:
                    # Convertir datetime en chaîne de date pour le regroupement
                    cash_deposits['date_str'] = cash_deposits['trx_date'].dt.strftime('%Y-%m-%d')
                    
                    # Grouper par chaîne de date
                    daily_deposits = cash_deposits.groupby('date_str').agg({
                        'branch': 'nunique',
                        'amount': 'sum'
                    }).reset_index()
                    
                    # Trouver les jours suspects
                    suspicious_days = daily_deposits[
                        (daily_deposits['branch'] > 1) & 
                        (daily_deposits['amount'] > 5000)
                    ]
                    
                    count = len(suspicious_days)
                    
                    # Collect the actual transactions that contributed to these suspicious days
                    if not suspicious_days.empty:
                        flagged_dates = suspicious_days['date_str'].tolist()
                        flagged_transactions = pd.concat([flagged_transactions, cash_deposits[cash_deposits['date_str'].isin(flagged_dates)]])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_suspected_money_mules(self, entity_id, transactions_df, wires_df, entity_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les mules financières suspectées basées sur les modèles de transaction.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        if transactions_df is not None and not transactions_df.empty:
            incoming_txns = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ].copy()
            
            if len(incoming_txns) > 0:
                deposit_txns = incoming_txns[incoming_txns['transaction_type_desc'] == 'Depot Especes']
                email_transfer_txns = incoming_txns[incoming_txns['transaction_type_desc'] == 'Transfert Internet']
                
                deposit_count = len(deposit_txns)
                email_transfer_count = len(email_transfer_txns)
                
                if entity_type == 'Particulier' and (deposit_count > 5 or email_transfer_count > 5):
                    count += 1
                    if deposit_count > 5:
                        flagged_transactions = pd.concat([flagged_transactions, deposit_txns])
                    if email_transfer_count > 5:
                        flagged_transactions = pd.concat([flagged_transactions, email_transfer_txns])
                
                outgoing_txns = transactions_df[
                    (transactions_df['party_key'] == entity_id) & 
                    (transactions_df['sign'] == '-')
                ].copy()
                
                if len(outgoing_txns) > 0 and len(incoming_txns) > 0:
                    incoming_txns = self._convert_dates(incoming_txns, 'trx_date')
                    outgoing_txns = self._convert_dates(outgoing_txns, 'trx_date')
                    
                    if not incoming_txns['trx_date'].empty and not outgoing_txns['trx_date'].empty:
                        # Regarder le premier dépôt et le dernier retrait
                        first_deposit_date = incoming_txns['trx_date'].min()
                        last_withdrawal_date = outgoing_txns['trx_date'].max()
                        
                        if last_withdrawal_date > first_deposit_date:
                            time_diff = (last_withdrawal_date - first_deposit_date).days
                            if time_diff <= 3:
                                count += 1
                                flagged_transactions = pd.concat([flagged_transactions, incoming_txns, outgoing_txns])
        
        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            
            if len(entity_wires) > 0 and entity_type == 'Particulier' and len(entity_wires) > 2:
                count += 1
                flagged_wires = pd.concat([flagged_wires, entity_wires])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_frequent_email_wire_transfers(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte l'utilisation fréquente de transferts par courriel et de virements internationaux.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if transactions_df is not None and not transactions_df.empty:
            email_transfers = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['transaction_type_desc'] == 'Transfert Internet')
            ].copy()
            count += len(email_transfers)
            if not email_transfers.empty:
                flagged_transactions = pd.concat([flagged_transactions, email_transfers])
        
        if wires_df is not None and not wires_df.empty:
            wires = wires_df[wires_df['party_key'] == entity_id].copy()
            count += len(wires)
            if not wires.empty:
                flagged_wires = pd.concat([flagged_wires, wires])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_mixed_funds_between_accounts(self, entity_id, transactions_df, wires_df, entity_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte le mélange de fonds entre différents comptes personnels et professionnels.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()
            
            if len(entity_txns) > 0 and 'account_type_desc' in entity_txns.columns:
                account_types = entity_txns['account_type_desc'].unique()
                
                if len(account_types) > 1:
                    count += 1
                    flagged_transactions = pd.concat([flagged_transactions, entity_txns])
                    
                    if entity_type == 'Particulier' and 'Entreprise' in account_types:
                        count += 2
                        flagged_transactions = pd.concat([flagged_transactions, entity_txns]) # Potentially duplicate, will be handled by final deduplication
        
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            
            if len(entity_wires) > 0:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()
                
                if unique_countries > 5:
                    count += 2
                    flagged_wires = pd.concat([flagged_wires, entity_wires])
                elif unique_countries > 3:
                    count += 1
                    flagged_wires = pd.concat([flagged_wires, entity_wires])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_high_volume_deposits(self, entity_id, transactions_df, wires_df, entity_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte un volume inhabituellement élevé de dépôts.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ].copy()
            
            count = len(deposits)
            if not deposits.empty:
                flagged_transactions = pd.concat([flagged_transactions, deposits])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_structured_deposits_below_threshold(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les dépôts structurés juste en dessous du seuil de déclaration.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) & 
                (transactions_df['sign'] == '+')
            ].copy()
            
            if len(deposits) > 0:
                structured_deposits = deposits[
                    (deposits['amount'] >= self.threshold_amount - self.margin) & 
                    (deposits['amount'] < self.threshold_amount)
                ]
                
                count = len(structured_deposits)
                if not structured_deposits.empty:
                    flagged_transactions = pd.concat([flagged_transactions, structured_deposits])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_quick_withdrawals_after_deposits(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les retraits rapides après les dépôts.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()
            
            if len(entity_txns) > 0:
                entity_txns = self._convert_dates(entity_txns, 'trx_date')
                
                deposits = entity_txns[entity_txns['sign'] == '+'].sort_values('trx_date')
                withdrawals = entity_txns[entity_txns['sign'] == '-'].sort_values('trx_date')
                
                if not deposits.empty and not withdrawals.empty:
                    for _, deposit in deposits.iterrows():
                        deposit_date = deposit['trx_date']
                        deposit_amount = deposit['amount']
                        
                        subsequent_withdrawals = withdrawals[
                            (withdrawals['trx_date'] > deposit_date) & 
                            (withdrawals['trx_date'] <= deposit_date + pd.Timedelta(days=3))
                        ]
                        
                        if not subsequent_withdrawals.empty:
                            withdrawal_amount = subsequent_withdrawals['amount'].sum()
                            
                            if withdrawal_amount >= 0.7 * deposit_amount:
                                count += 1
                                flagged_transactions = pd.concat([flagged_transactions, deposit.to_frame().T, subsequent_withdrawals])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_foreign_exchange_wires(self, entity_id, transactions_df, wires_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte les gros virements provenant de sociétés de change.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            
            if len(entity_wires) > 0:
                forex_wires = entity_wires[
                    entity_wires['originator'].str.contains('change|forex|exchange|money', case=False, na=False)
                ]
                
                count = len(forex_wires)
                if not forex_wires.empty:
                    flagged_wires = pd.concat([flagged_wires, forex_wires])
        
        return count, flagged_transactions, flagged_wires
    
    def detect_inconsistent_activity(self, entity_id, transactions_df, wires_df, entity_df) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Détecte une activité incohérente avec le profil ou le type d'entreprise.
        """
        count = 0
        flagged_transactions = pd.DataFrame()
        flagged_wires = pd.DataFrame()
        
        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_type = entity_info['account_type_desc'].iloc[0]
        
        if transactions_df is not None and not transactions_df.empty and entity_type is not None:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()
            
            if len(entity_txns) > 0:
                if entity_type == 'Particulier':
                    commercial_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Entreprise|Commercial|Business', case=False, na=False)
                    ]
                    
                    count += len(commercial_txns)
                    if not commercial_txns.empty:
                        flagged_transactions = pd.concat([flagged_transactions, commercial_txns])
                
                elif entity_type == 'Entreprise':
                    personal_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Personnel|Personal', case=False, na=False)
                    ]
                    
                    count += len(personal_txns)
                    if not personal_txns.empty:
                        flagged_transactions = pd.concat([flagged_transactions, personal_txns])
        
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            
            if len(entity_wires) > 0:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()
                
                if unique_countries > 5:
                    count += 2
                    if not entity_wires.empty: # Flag all wires for this entity if rule triggers significantly
                        flagged_wires = pd.concat([flagged_wires, entity_wires])
                elif unique_countries > 3:
                    count += 1
                    if not entity_wires.empty: # Flag all wires for this entity if rule triggers significantly
                        flagged_wires = pd.concat([flagged_wires, entity_wires])
        
        return count, flagged_transactions, flagged_wires
    
    def apply_prior_suspicious_flag_boost(self, entity_id, entity_df):
        """
        Applique un boost de score si l'entité a été précédemment signalée comme suspecte.
        """
        if entity_df is not None and not entity_df.empty:
            if 'prior_suspicious_flag' in entity_df.columns:
                entity_info = entity_df[entity_df['party_key'] == entity_id]
                if not entity_info.empty and entity_info['prior_suspicious_flag'].iloc[0] == 1:
                    return self.prior_suspicious_flag_boost
        
        return 0
    
    def get_rule_descriptions(self):
        """
        Retourne les descriptions des règles pour l'explication des alertes.
        """
        descriptions = {
            "large_wire_transfers_followed_by_outgoing": "Reçoit soudainement des virements importants suivis de transferts sortants",
            "sanctioned_countries_transactions": "Transactions avec des pays sanctionnés ou à haut risque",
            "split_cash_deposits_same_day": "Dépôts en espèces fractionnés le même jour à plusieurs endroits",
            "suspected_money_mules": "Utilisation de mules financières suspectées",
            "frequent_email_wire_transfers": "Utilisation fréquente de transferts par courriel et de virements internationaux",
            "mixed_funds_between_accounts": "Mélange de fonds entre comptes personnels et professionnels",
            "high_volume_deposits": "Volume inhabituellement élevé de dépôts",
            "structured_deposits_below_threshold": "Dépôts structurés sous le seuil de déclaration de 10 000$",
            "quick_withdrawals_after_deposits": "Retraits rapides après les dépôts",
            "foreign_exchange_wires": "Gros virements provenant de sociétés de change",
            "inconsistent_activity": "Activité incohérente avec le profil ou le type d'entreprise",
            "prior_suspicious_flag_boost": "Compte précédemment signalé comme suspect"
        }
        
        return descriptions
    
    def calculate_score(self, entity_id, transactions_df, wires_df, entity_df) -> Tuple[int, Dict, pd.DataFrame, pd.DataFrame]:
        """
        Calcule le score global basé sur les règles pour une entité et retourne les règles déclenchées et les transactions/virements marqués.
        """
        score = 0
        triggered_rules = {}
        all_flagged_transactions = []
        all_flagged_wires = []
        
        # Appliquer chaque règle, ajouter les scores et enregistrer les règles déclenchées
        rule_score_val, relevant_txns, relevant_wires = self.detect_large_wire_transfers_followed_by_outgoing(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("large_wire_transfers_followed_by_outgoing", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["large_wire_transfers_followed_by_outgoing"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_sanctioned_countries_transactions(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("sanctioned_countries_transactions", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["sanctioned_countries_transactions"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_split_cash_deposits_same_day(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("split_cash_deposits_same_day", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["split_cash_deposits_same_day"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_suspected_money_mules(entity_id, transactions_df, wires_df, entity_df)
        rule_score = self._get_graduated_score("suspected_money_mules", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["suspected_money_mules"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_frequent_email_wire_transfers(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("frequent_email_wire_transfers", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["frequent_email_wire_transfers"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_mixed_funds_between_accounts(entity_id, transactions_df, wires_df, entity_df)
        rule_score = self._get_graduated_score("mixed_funds_between_accounts", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["mixed_funds_between_accounts"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_high_volume_deposits(entity_id, transactions_df, wires_df, entity_df)
        rule_score = self._get_graduated_score("high_volume_deposits", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["high_volume_deposits"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_structured_deposits_below_threshold(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("structured_deposits_below_threshold", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["structured_deposits_below_threshold"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_quick_withdrawals_after_deposits(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("quick_withdrawals_after_deposits", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["quick_withdrawals_after_deposits"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_foreign_exchange_wires(entity_id, transactions_df, wires_df)
        rule_score = self._get_graduated_score("foreign_exchange_wires", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["foreign_exchange_wires"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)

        rule_score_val, relevant_txns, relevant_wires = self.detect_inconsistent_activity(entity_id, transactions_df, wires_df, entity_df)
        rule_score = self._get_graduated_score("inconsistent_activity", rule_score_val)
        score += rule_score
        if rule_score > 0:
            triggered_rules["inconsistent_activity"] = rule_score
            if relevant_txns is not None and not relevant_txns.empty:
                all_flagged_transactions.append(relevant_txns)
            if relevant_wires is not None and not relevant_wires.empty:
                all_flagged_wires.append(relevant_wires)
        
        # Appliquer le boost de drapeau suspect antérieur
        prior_flag_boost = self.apply_prior_suspicious_flag_boost(entity_id, entity_df)
        if prior_flag_boost > 0:
            score += prior_flag_boost
            triggered_rules["prior_suspicious_flag_boost"] = prior_flag_boost
        
        # Plafonner au score maximum
        normalized_score = min(score, self.max_score)
        
        # Concaténer et dédupliquer les transactions et virements marqués
        final_flagged_transactions_df = pd.DataFrame()
        if all_flagged_transactions:
            final_flagged_transactions_df = pd.concat(all_flagged_transactions, ignore_index=True)
            # Assuming 'transaction_id' is a unique identifier, or use a combination of columns
            # to ensure unique identification of transactions that triggered rules.
            # For now, drop duplicates based on all columns, but ideally would use unique IDs.
            if 'transaction_id' in final_flagged_transactions_df.columns:
                final_flagged_transactions_df = final_flagged_transactions_df.drop_duplicates(subset=['transaction_id'])
            else:
                final_flagged_transactions_df = final_flagged_transactions_df.drop_duplicates()

        final_flagged_wires_df = pd.DataFrame()
        if all_flagged_wires:
            final_flagged_wires_df = pd.concat(all_flagged_wires, ignore_index=True)
            # Assuming 'wire_id' is a unique identifier
            if 'wire_id' in final_flagged_wires_df.columns:
                final_flagged_wires_df = final_flagged_wires_df.drop_duplicates(subset=['wire_id'])
            else:
                final_flagged_wires_df = final_flagged_wires_df.drop_duplicates()

        return normalized_score, triggered_rules, final_flagged_transactions_df, final_flagged_wires_df 