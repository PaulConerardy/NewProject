import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import yaml
from detection_rules import (
    DetectionRule,
    LargeWireTransferRule,
    SanctionedCountriesRule,
    SplitCashDepositsRule,
    SuspectedMoneyMulesRule,
    FrequentEmailWireTransfersRule,
    MixedFundsBetweenAccountsRule,
    HighVolumeDepositsRule,
    StructuredDepositsRule,
    QuickWithdrawalsRule,
    ForeignExchangeWiresRule,
    InconsistentActivityRule
) # Import all modular rule classes


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
                'suspicious_countries_list': ['CN', 'HK', 'AE', 'IR', 'KW'],
                'threshold_amount': 10000,
                'margin': 1000,
                'max_score': 100,
                'large_amount_threshold': 5000,
                'subsequent_transfer_count_threshold': 3,
                'subsequent_transfer_time_window_days': 3,
                'split_deposit_daily_amount_threshold': 5000,
                'split_deposit_min_branches': 1,
                'mule_individual_deposit_threshold': 5, # Added for SuspectedMoneyMulesRule
                'mule_individual_email_transfer_threshold': 5, # Added for SuspectedMoneyMulesRule
                'mule_withdrawal_time_window_days': 3, # Added for SuspectedMoneyMulesRule
                'mule_individual_wire_threshold': 2, # Added for SuspectedMoneyMulesRule
                'quick_withdrawal_time_window_days': 3,
                'quick_withdrawal_percentage_threshold': 0.7,
                'forex_keywords': 'change|forex|exchange|money',
                'inconsistent_activity_multi_country_threshold_high': 5,
                'inconsistent_activity_multi_country_threshold_low': 3
            }
        }
        
        # Charger la configuration si fournie
        if config_path:
            with open(config_path, 'r') as file:
                loaded_config = yaml.safe_load(file)
                # Merge default config with loaded config (handles missing keys in file)
                self.config = {**self.config, **loaded_config}
                # Ensure nested dicts are also merged/updated appropriately
                self.config['rules'] = {**self.config.get('rules', {}), **loaded_config.get('rules', {})}
                self.config['detection_params'] = {**self.config.get('detection_params', {}), **loaded_config.get('detection_params', {}) }
        
        # Définir les paramètres de détection
        self.alert_threshold = self.config['alert_threshold']
        self.prior_suspicious_flag_boost = self.config['prior_suspicious_flag_boost']
        self.max_score = self.config['max_score']
        
        # Initialize modular rules
        self.rules: List[DetectionRule] = [
            LargeWireTransferRule(self.config),
            SanctionedCountriesRule(self.config),
            SplitCashDepositsRule(self.config),
            SuspectedMoneyMulesRule(self.config),
            FrequentEmailWireTransfersRule(self.config),
            MixedFundsBetweenAccountsRule(self.config),
            HighVolumeDepositsRule(self.config),
            StructuredDepositsRule(self.config),
            QuickWithdrawalsRule(self.config),
            ForeignExchangeWiresRule(self.config),
            InconsistentActivityRule(self.config),
        ]
    
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
    
    def get_rule_details(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Retourne les détails des règles déclenchées pour une entité spécifique.
        """
        triggered_rules_with_scores = {}
        all_details = {}

        for rule in self.rules:
            score, details = rule.apply(entity_id, transactions_df, wires_df, entity_df)
            if score > 0:
                triggered_rules_with_scores[rule.name] = score
                if details:
                    all_details[rule.name] = details

        # Check for prior suspicious flag boost separately if not integrated as a rule
        prior_flag_boost = self.apply_prior_suspicious_flag_boost(entity_id, entity_df)
        if prior_flag_boost > 0:
             if "prior_suspicious_flag_boost" not in triggered_rules_with_scores: # Avoid overwriting if it's somehow a rule name
                  triggered_rules_with_scores["prior_suspicious_flag_boost"] = prior_flag_boost
             # No specific details added for the boost itself here unless needed

        # Note: The returned triggered_rules_with_scores only contain rule names and scores.
        # The detailed breakdown per rule is in all_details but not returned by this method currently.
        # The Alert class uses the keys of triggered_rules_with_scores to summarize patterns.
        # Consider if all_details should be returned or used differently.

        return triggered_rules_with_scores # Only returning rule names and scores for compatibility with Alert class
    
    def get_rule_descriptions(self) -> Dict[str, str]:
        """
        Retourne les descriptions des règles pour l'explication des alertes.
        """
        descriptions = {
            rule.name: rule.get_description() for rule in self.rules
        }
        # Add description for the prior suspicious flag boost if not a modular rule
        if "prior_suspicious_flag_boost" not in descriptions:
             descriptions["prior_suspicious_flag_boost"] = "Compte précédemment signalé comme suspect"

        return descriptions
    
    def calculate_score(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Calcule le score global basé sur les règles pour une entité.
        """
        total_score = 0
        triggered_rules_with_scores = {}
        # Note: Details per rule are calculated but not aggregated or returned here currently.
        # If rule-specific details are needed downstream (e.g., in Alert summary),
        # this method or get_rule_details needs to be adjusted.

        for rule in self.rules:
            score, details = rule.apply(entity_id, transactions_df, wires_df, entity_df)
            if score > 0:
                total_score += score
                triggered_rules_with_scores[rule.name] = score # Storing score here for consistency
                # Details from individual rules are not currently stored or returned by this method

        # Apply prior suspicious flag boost
        prior_flag_boost = self.apply_prior_suspicious_flag_boost(entity_id, entity_df)
        if prior_flag_boost > 0:
            total_score += prior_flag_boost
            if "prior_suspicious_flag_boost" not in triggered_rules_with_scores: # Avoid adding twice if rule exists
                 triggered_rules_with_scores["prior_suspicious_flag_boost"] = prior_flag_boost # Store boost as if it's a rule score

        # Plafonner au score maximum
        normalized_score = min(total_score, self.max_score)

        # The triggered_rules_with_scores dict here only contains rule names and their *individual* scores.
        # The Alert class summarizes patterns based on the *names* (keys) in this dict.
        # If the summary in Alert needs the individual rule scores, this is fine.
        # If it needs the details (e.g., list of transactions), calculate_score or get_rule_details
        # would need to return/handle all_details.

        return normalized_score, triggered_rules_with_scores # Returning dict with scores for potential future use 