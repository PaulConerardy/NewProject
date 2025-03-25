import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.window import Window
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
        # Initialiser Spark
        self.spark = SparkSession.builder \
            .appName("AML Detection") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
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
            df (DataFrame): DataFrame contenant la colonne de date
            date_column (str): Nom de la colonne contenant les dates
            
        Returns:
            DataFrame: DataFrame avec les dates converties
        """
        # Tenter de convertir le format de date
        try:
            # Essayer avec le format spécifique d'abord
            df = df.withColumn(date_column, 
                F.to_timestamp(F.col(date_column), 'ddMMMyyyy'))
            
            # Vérifier si toutes les dates sont NULL
            if df.filter(F.col(date_column).isNotNull()).count() == 0:
                # Si toutes les dates sont NULL, essayer sans format spécifique
                df = df.withColumn(date_column, 
                    F.to_timestamp(F.col(date_column)))
        except Exception:
            # Si la conversion échoue, essayer sans format spécifique
            df = df.withColumn(date_column, 
                F.to_timestamp(F.col(date_column)))
        
        return df
    
    def detect_large_wire_transfers_followed_by_outgoing(self, entity_id, transactions_df, wires_df):
        """
        Détecte les entités qui reçoivent soudainement des virements importants ou des dépôts
        en espèces suivis d'une augmentation des transferts sortants.
        """
        count = 0
        
        if wires_df is not None and wires_df.count() > 0:
            # Filtrer les virements entrants pour cette entité
            incoming_wires = wires_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("sign") == "+")
            )
            
            if incoming_wires.count() > 0:
                # Identifier les gros virements (plus de 5000$)
                large_wires = incoming_wires.filter(F.col("amount") > 5000)
                
                if large_wires.count() > 0:
                    count += 1
                    
                    # Vérifier les transactions sortantes après les dates des gros virements
                    if transactions_df is not None and transactions_df.count() > 0:
                        large_wires = self._convert_dates(large_wires, 'wire_date')
                        
                        if large_wires.count() > 0:
                            min_date = large_wires.agg(F.min("wire_date")).collect()[0][0]
                            
                            transactions_df = self._convert_dates(transactions_df, 'trx_date')

                            outgoing_txns = transactions_df.filter(
                                (F.col("party_key") == entity_id) & 
                                (F.col("sign") == "-") &
                                (F.col("trx_date") > min_date)
                            )
                            
                            if outgoing_txns.count() > 3:
                                count += 1
                                
                                unique_types = outgoing_txns.select("transaction_type_desc").distinct().count()
                                if unique_types > 1:
                                    count += 1
        
        # Vérifier aussi les gros dépôts en espèces
        if transactions_df is not None and transactions_df.count() > 0:
            transactions_df = self._convert_dates(transactions_df, 'trx_date')

            cash_deposits = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("transaction_type_desc") == "Depot Especes") &
                (F.col("sign") == "+")
            )
            
            if cash_deposits.count() > 0:
                large_deposits = cash_deposits.filter(F.col("amount") > 5000)
                if large_deposits.count() > 0:
                    count += 1
                    
                if large_deposits.count() > 0:
                    min_date = large_deposits.agg(F.min("trx_date")).collect()[0][0]
                    
                    outgoing_txns = transactions_df.filter(
                        (F.col("party_key") == entity_id) & 
                        (F.col("sign") == "-") &
                        (F.col("trx_date") > min_date)
                    )
                    
                    if outgoing_txns.count() > 3:
                        count += 1
        
        return self._get_graduated_score('large_wire_transfers_followed_by_outgoing', count)
    
    def detect_sanctioned_countries_transactions(self, entity_id, transactions_df, wires_df):
        """
        Détecte les transactions avec des pays sanctionnés ou à haut risque.
        """
        count = 0
        
        if wires_df is not None and wires_df.count() > 0:
            entity_wires = wires_df.filter(F.col("party_key") == entity_id)
            
            if entity_wires.count() > 0:
                suspicious_origin = entity_wires.filter(
                    F.col("originator_country").isin(["CN", "HK", "AE", "IR", "KW"])
                )
                suspicious_dest = entity_wires.filter(
                    F.col("beneficiary_country").isin(["CN", "HK", "AE", "IR", "KW"])
                )
                
                count = suspicious_origin.count() + suspicious_dest.count()
        
        return self._get_graduated_score('sanctioned_countries_transactions', count)
    
    def detect_split_cash_deposits_same_day(self, entity_id, transactions_df, wires_df):
        """
        Détecte les dépôts en espèces fractionnés le même jour.
        """
        count = 0
        
        if transactions_df is not None and transactions_df.count() > 0:
            # Filtrer les dépôts en espèces
            cash_deposits = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("transaction_type_desc") == "Depot Especes") &
                (F.col("sign") == "+")
            )
            
            if cash_deposits.count() > 0:
                # Convertir les dates
                cash_deposits = self._convert_dates(cash_deposits, 'trx_date')
                
                # Ajouter colonne de chaîne de date pour le regroupement
                cash_deposits = cash_deposits.withColumn(
                    "date_str", 
                    F.date_format(F.col("trx_date"), "yyyy-MM-dd")
                )
                
                # Grouper par chaîne de date
                daily_deposits = cash_deposits.groupBy("date_str").agg(
                    F.countDistinct("branch").alias("branch_count"),
                    F.sum("amount").alias("total_amount")
                )
                
                # Trouver les jours suspects
                suspicious_days = daily_deposits.filter(
                    (F.col("branch_count") > 1) & 
                    (F.col("total_amount") > 5000)
                )
                
                count = suspicious_days.count()
        
        return self._get_graduated_score('split_cash_deposits_same_day', count)
    
    def detect_suspected_money_mules(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte les mules financières suspectées basées sur les modèles de transaction.
        """
        count = 0
        
        entity_type = None
        if entity_df is not None and entity_df.count() > 0:
            entity_info = entity_df.filter(F.col("party_key") == entity_id)
            if entity_info.count() > 0:
                entity_type = entity_info.select("account_type_desc").first()[0]
        
        if transactions_df is not None and transactions_df.count() > 0:
            incoming_txns = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("sign") == "+")
            )
            
            if incoming_txns.count() > 0:
                deposit_count = incoming_txns.filter(
                    F.col("transaction_type_desc") == "Depot Especes"
                ).count()
                email_transfer_count = incoming_txns.filter(
                    F.col("transaction_type_desc") == "Transfert Internet"
                ).count()
                
                if entity_type == "Particulier" and (deposit_count > 5 or email_transfer_count > 5):
                    count += 1
                
                outgoing_txns = transactions_df.filter(
                    (F.col("party_key") == entity_id) & 
                    (F.col("sign") == "-")
                )
                
                if outgoing_txns.count() > 0 and incoming_txns.count() > 0:
                    incoming_txns = self._convert_dates(incoming_txns, 'trx_date')
                    outgoing_txns = self._convert_dates(outgoing_txns, 'trx_date')
                    
                    # Regarder le premier dépôt et le dernier retrait
                    first_deposit = incoming_txns.agg(F.min("trx_date")).first()
                    last_withdrawal = outgoing_txns.agg(F.max("trx_date")).first()
                    
                    if first_deposit[0] is not None and last_withdrawal[0] is not None:
                        first_deposit_date = first_deposit[0]
                        last_withdrawal_date = last_withdrawal[0]
                        
                        if last_withdrawal_date > first_deposit_date:
                            # Calculer la différence en jours
                            time_diff_seconds = (last_withdrawal_date.timestamp() - first_deposit_date.timestamp())
                            time_diff_days = time_diff_seconds / (60 * 60 * 24)
                            if time_diff_days <= 3:
                                count += 1
        
        if wires_df is not None and wires_df.count() > 0:
            entity_wires = wires_df.filter(F.col("party_key") == entity_id)
            
            if entity_wires.count() > 0 and entity_type == "Particulier" and entity_wires.count() > 2:
                count += 1
        
        return self._get_graduated_score('suspected_money_mules', count)
    
    def detect_frequent_email_wire_transfers(self, entity_id, transactions_df, wires_df):
        """
        Détecte l'utilisation fréquente de transferts par courriel et de virements internationaux.
        """
        count = 0
        
        if transactions_df is not None and transactions_df.count() > 0:
            email_transfers = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("transaction_type_desc") == "Transfert Internet")
            )
            count += email_transfers.count()
        
        if wires_df is not None and wires_df.count() > 0:
            wires = wires_df.filter(F.col("party_key") == entity_id)
            count += wires.count()
        
        return self._get_graduated_score('frequent_email_wire_transfers', count)
    
    def detect_mixed_funds_between_accounts(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte le mélange de fonds entre différents comptes personnels et professionnels.
        """
        count = 0
        
        entity_type = None
        if entity_df is not None and entity_df.count() > 0:
            entity_info = entity_df.filter(F.col("party_key") == entity_id)
            if entity_info.count() > 0:
                entity_type = entity_info.select("account_type_desc").first()[0]
        
        if transactions_df is not None and transactions_df.count() > 0:
            entity_txns = transactions_df.filter(F.col("party_key") == entity_id)
            
            if entity_txns.count() > 0 and "account_type_desc" in entity_txns.columns:
                account_types = [row[0] for row in entity_txns.select("account_type_desc").distinct().collect()]
                
                if len(account_types) > 1:
                    count += 1
                    
                    if entity_type == "Particulier" and "Entreprise" in account_types:
                        count += 2
        
        return self._get_graduated_score('mixed_funds_between_accounts', count)
    
    def detect_high_volume_deposits(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte un volume inhabituellement élevé de dépôts.
        """
        count = 0
        
        if transactions_df is not None and transactions_df.count() > 0:
            deposits = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("sign") == "+")
            )
            
            count = deposits.count()
        
        return self._get_graduated_score('high_volume_deposits', count)
    
    def detect_structured_deposits_below_threshold(self, entity_id, transactions_df, wires_df):
        """
        Détecte les dépôts structurés juste en dessous du seuil de déclaration.
        """
        count = 0
        
        if transactions_df is not None and transactions_df.count() > 0:
            deposits = transactions_df.filter(
                (F.col("party_key") == entity_id) & 
                (F.col("sign") == "+")
            )
            
            if deposits.count() > 0:
                structured_deposits = deposits.filter(
                    (F.col("amount") >= self.threshold_amount - self.margin) & 
                    (F.col("amount") < self.threshold_amount)
                )
                
                count = structured_deposits.count()
        
        return self._get_graduated_score('structured_deposits_below_threshold', count)
    
    def detect_quick_withdrawals_after_deposits(self, entity_id, transactions_df, wires_df):
        """
        Détecte les retraits rapides après les dépôts.
        """
        count = 0
        
        if transactions_df is not None and transactions_df.count() > 0:
            entity_txns = transactions_df.filter(F.col("party_key") == entity_id)
            
            if entity_txns.count() > 0:
                entity_txns = self._convert_dates(entity_txns, 'trx_date')
                
                # Pour cette règle, nous avons besoin de collecter et comparer manuellement
                # car PySpark ne supporte pas facilement les opérations itératives comme pandas
                if entity_txns.filter(F.col("trx_date").isNotNull()).count() > 0:
                    deposits = entity_txns.filter(F.col("sign") == "+").orderBy("trx_date")
                    withdrawals = entity_txns.filter(F.col("sign") == "-").orderBy("trx_date")
                    
                    if deposits.count() > 0 and withdrawals.count() > 0:
                        # Collectons les dépôts et retraits pour les traiter localement
                        deposit_list = deposits.select("trx_date", "amount").collect()
                        withdrawal_df = withdrawals.select("trx_date", "amount")
                        
                        for deposit in deposit_list:
                            deposit_date = deposit["trx_date"]
                            deposit_amount = deposit["amount"]
                            
                            if deposit_date is not None:
                                # Calculer la date limite (3 jours après le dépôt)
                                three_days_later = deposit_date.timestamp() + (3 * 24 * 60 * 60)
                                three_days_later_date = datetime.fromtimestamp(three_days_later)
                                
                                # Filtrer les retraits de cette période
                                subsequent_withdrawals = withdrawal_df.filter(
                                    (F.col("trx_date") > deposit_date) & 
                                    (F.col("trx_date") <= F.lit(three_days_later_date))
                                )
                                
                                if subsequent_withdrawals.count() > 0:
                                    withdrawal_amount = subsequent_withdrawals.agg(F.sum("amount")).first()[0]
                                    
                                    if withdrawal_amount >= 0.7 * deposit_amount:
                                        count += 1
        
        return self._get_graduated_score('quick_withdrawals_after_deposits', count)
    
    def detect_foreign_exchange_wires(self, entity_id, transactions_df, wires_df):
        """
        Détecte les gros virements provenant de sociétés de change.
        """
        count = 0
        
        if wires_df is not None and wires_df.count() > 0:
            entity_wires = wires_df.filter(F.col("party_key") == entity_id)
            
            if entity_wires.count() > 0:
                forex_wires = entity_wires.filter(
                    F.lower(F.col("originator")).rlike("change|forex|exchange|money")
                )
                
                count = forex_wires.count()
        
        return self._get_graduated_score('foreign_exchange_wires', count)
    
    def detect_inconsistent_activity(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Détecte une activité incohérente avec le profil ou le type d'entreprise.
        """
        count = 0
        
        entity_type = None
        if entity_df is not None and entity_df.count() > 0:
            entity_info = entity_df.filter(F.col("party_key") == entity_id)
            if entity_info.count() > 0:
                entity_type = entity_info.select("account_type_desc").first()[0]
        
        if transactions_df is not None and transactions_df.count() > 0 and entity_type is not None:
            entity_txns = transactions_df.filter(F.col("party_key") == entity_id)
            
            if entity_txns.count() > 0:
                if entity_type == "Particulier":
                    commercial_txns = entity_txns.filter(
                        F.lower(F.col("transaction_type_desc")).rlike("entreprise|commercial|business")
                    )
                    
                    count += commercial_txns.count()
                
                elif entity_type == "Entreprise":
                    personal_txns = entity_txns.filter(
                        F.lower(F.col("transaction_type_desc")).rlike("personnel|personal")
                    )
                    
                    count += personal_txns.count()
        
        if wires_df is not None and wires_df.count() > 0 and entity_type == "Particulier":
            entity_wires = wires_df.filter(F.col("party_key") == entity_id)
            
            if entity_wires.count() > 0:
                # Collect all unique countries
                originator_countries = [
                    row[0] for row in entity_wires.select("originator_country")
                    .filter(F.col("originator_country").isNotNull())
                    .distinct().collect()
                ]
                
                beneficiary_countries = [
                    row[0] for row in entity_wires.select("beneficiary_country")
                    .filter(F.col("beneficiary_country").isNotNull())
                    .distinct().collect()
                ]
                
                unique_countries = set(originator_countries + beneficiary_countries)
                
                if len(unique_countries) > 5:
                    count += 2
                elif len(unique_countries) > 3:
                    count += 1
        
        return self._get_graduated_score('inconsistent_activity', count)
    
    def apply_prior_suspicious_flag_boost(self, entity_id, entity_df):
        """
        Applique un boost de score si l'entité a été précédemment signalée comme suspecte.
        """
        if entity_df is not None and entity_df.count() > 0:
            if "prior_suspicious_flag" in entity_df.columns:
                entity_info = entity_df.filter(F.col("party_key") == entity_id)
                if entity_info.count() > 0 and entity_info.select("prior_suspicious_flag").first()[0] == 1:
                    return self.prior_suspicious_flag_boost
        
        return 0
    
    def get_rule_details(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Retourne les détails des règles déclenchées pour une entité spécifique.
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
        
        # Ajouter le boost de drapeau suspect antérieur si applicable
        prior_flag_boost = self.apply_prior_suspicious_flag_boost(entity_id, entity_df)
        if prior_flag_boost > 0:
            rule_details["prior_suspicious_flag_boost"] = prior_flag_boost
        
        # Filtrer pour ne garder que les règles avec un score > 0
        triggered_rules = {rule: score for rule, score in rule_details.items() if score > 0}
        
        return triggered_rules
    
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
    
    def calculate_score(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Calcule le score global basé sur les règles pour une entité.
        """
        score = 0
        
        # Appliquer chaque règle et ajouter les scores
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
        
        # Appliquer le boost de drapeau suspect antérieur
        score += self.apply_prior_suspicious_flag_boost(entity_id, entity_df)
        
        # Obtenir les détails des règles déclenchées
        triggered_rules = self.get_rule_details(entity_id, transactions_df, wires_df, entity_df)
        
        # Plafonner au score maximum
        normalized_score = min(score, self.max_score)
        return normalized_score, triggered_rules 