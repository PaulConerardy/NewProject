import os
import pandas as pd
import numpy as np
import yaml
import json
from rule_based_detection import RuleBasedDetection
from visualization import Visualizer

class DetectionRunner:
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le Runner de Détection avec la configuration.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        # Charger la configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialiser le composant de détection basé sur les règles
        self.detector = RuleBasedDetection(config_path)
        
        # Définir les chemins pour les fichiers d'entrée et de sortie
        self.input_paths = self.config['input_paths']
        self.output_paths = self.config['output_paths']
        
        # Créer les répertoires de sortie s'ils n'existent pas
        self._create_output_dirs()
        
        # Définir le seuil d'alerte
        self.alert_threshold = self.config['alert_threshold']
        
        # Initialiser le visualiseur
        self.visualizer = Visualizer()
        
    def _create_output_dirs(self):
        """Créer les répertoires de sortie s'ils n'existent pas."""
        for path in self.output_paths.values():
            output_dir = os.path.dirname(path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
    def load_data(self):
        """
        Charger les données depuis les fichiers d'entrée.
        
        Returns:
            tuple: (entity_df, transactions_df, wires_df)
        """
        entity_df = None
        transactions_df = None
        wires_df = None
        
        # Charger les données des entités
        if 'entities' in self.input_paths and os.path.exists(self.input_paths['entities']):
            entity_df = pd.read_csv(self.input_paths['entities'])
            # Ajouter la colonne prior_suspicious_flag si elle n'existe pas
            if 'prior_suspicious_flag' not in entity_df.columns:
                entity_df['prior_suspicious_flag'] = 0
        
        # Charger les données des transactions
        if 'transactions' in self.input_paths and os.path.exists(self.input_paths['transactions']):
            transactions_df = pd.read_csv(self.input_paths['transactions'])
        
        # Charger les données des virements
        if 'wires' in self.input_paths and os.path.exists(self.input_paths['wires']):
            wires_df = pd.read_csv(self.input_paths['wires'])
        
        return entity_df, transactions_df, wires_df
    
    def run_detection(self, entity_df, transactions_df, wires_df):
        """
        Exécuter la détection sur les données chargées.
        
        Args:
            entity_df (DataFrame): Données des entités.
            transactions_df (DataFrame): Données des transactions.
            wires_df (DataFrame): Données des virements.
            
        Returns:
            tuple: (alerted_accounts_df, flagged_transactions_df)
        """
        # Préparer les DataFrames de résultats
        alerted_accounts = []
        all_flagged_transactions_for_saving = []
        
        # Obtenir les IDs uniques des entités
        if entity_df is not None and not entity_df.empty:
            entity_ids = entity_df['party_key'].unique()
        elif transactions_df is not None and not transactions_df.empty:
            entity_ids = transactions_df['party_key'].unique()
        elif wires_df is not None and not wires_df.empty:
            entity_ids = wires_df['party_key'].unique()
        else:
            return pd.DataFrame(), pd.DataFrame()
        
        # Traiter chaque entité
        for entity_id in entity_ids:
            # Calculer le score pour l'entité
            score, triggered_rules, entity_flagged_txns, entity_flagged_wires = self.detector.calculate_score(entity_id, transactions_df, wires_df, entity_df)
            
            # Si le score est supérieur au seuil, alerter le compte
            if score >= self.alert_threshold:
                # Obtenir les détails de l'entité
                entity_info = {}
                if entity_df is not None and not entity_df.empty:
                    entity_info = entity_df[entity_df['party_key'] == entity_id].iloc[0].to_dict()
                
                # Ajouter aux comptes alertés
                account_record = {
                    'party_key': entity_id,
                    'account_key': entity_info.get('account_key', None) if entity_info else None,
                    'total_score': score,
                    'triggered_rules': json.dumps(triggered_rules),
                    'prior_suspicious_flag': entity_info.get('prior_suspicious_flag', 0) if entity_info else 0
                }
                alerted_accounts.append(account_record)
                
                # Marquer les transactions et virements qui ont été identifiés par les règles
                current_entity_flagged_data = []
                if not entity_flagged_txns.empty:
                    current_entity_flagged_data.append(entity_flagged_txns)
                if not entity_flagged_wires.empty:
                    current_entity_flagged_data.append(entity_flagged_wires)

                if current_entity_flagged_data:
                    combined_flagged_data = pd.concat(current_entity_flagged_data, ignore_index=True)
                    combined_flagged_data['is_flagged'] = 1
                    combined_flagged_data['total_score'] = score

                    # Add rule scores to the flagged transactions/wires
                    for rule, rule_score in triggered_rules.items():
                        # Only add rule columns that are actually present in the data
                        if rule in self.detector.get_rule_descriptions(): # Check if it's a known rule
                             combined_flagged_data[f'rule_{rule}'] = rule_score
                    
                    all_flagged_transactions_for_saving.append(combined_flagged_data)

        
        # Créer les DataFrames finaux
        alerted_accounts_df = pd.DataFrame(alerted_accounts) if alerted_accounts else pd.DataFrame()
        flagged_transactions_df = pd.concat(all_flagged_transactions_for_saving) if all_flagged_transactions_for_saving else pd.DataFrame()

        # Drop duplicates based on a combination of identifying columns, adjust as needed for your data schema
        # For transactions, 'party_key', 'account_key', 'trx_date', 'amount', 'transaction_id' might be relevant
        # For wires, 'party_key', 'wire_date', 'amount', 'wire_id' might be relevant
        # Given that we are concatenating both transactions and wires, a general approach is needed.
        # If 'transaction_id' and 'wire_id' exist, use them. Otherwise, use all columns.
        if 'transaction_id' in flagged_transactions_df.columns and 'wire_id' in flagged_transactions_df.columns:
            flagged_transactions_df = flagged_transactions_df.drop_duplicates(subset=['transaction_id', 'wire_id'])
        elif 'transaction_id' in flagged_transactions_df.columns:
            flagged_transactions_df = flagged_transactions_df.drop_duplicates(subset=['transaction_id'])
        elif 'wire_id' in flagged_transactions_df.columns:
            flagged_transactions_df = flagged_transactions_df.drop_duplicates(subset=['wire_id'])
        else:
            # Fallback if no specific IDs are available, might be less precise
            flagged_transactions_df = flagged_transactions_df.drop_duplicates()

        
        return alerted_accounts_df, flagged_transactions_df
    
    def save_results(self, alerted_accounts_df, flagged_transactions_df):
        """
        Sauvegarder les résultats de détection dans les fichiers de sortie.
        
        Args:
            alerted_accounts_df (DataFrame): Données des comptes alertés.
            flagged_transactions_df (DataFrame): Données des transactions marquées.
        """
        if not alerted_accounts_df.empty:
            alerted_accounts_df.to_csv(self.output_paths['alerted_accounts'], index=False)
            print(f"Comptes alertés sauvegardés dans {self.output_paths['alerted_accounts']}")
        else:
            print("Aucun compte n'a été alerté.")
        
        if not flagged_transactions_df.empty:
            flagged_transactions_df.to_csv(self.output_paths['flagged_transactions'], index=False)
            print(f"Transactions marquées sauvegardées dans {self.output_paths['flagged_transactions']}")
        else:
            print("Aucune transaction n'a été marquée.")
        
        # Générer les visualisations
        if not alerted_accounts_df.empty:
            print("Génération des visualisations...")
            self.visualizer.visualize_detection_results(alerted_accounts_df, flagged_transactions_df)
            print(f"Visualisations sauvegardées dans {self.visualizer.output_dir}")
    
    def run(self):
        """Méthode principale d'exécution pour lancer le processus de détection."""
        print("Chargement des données...")
        entity_df, transactions_df, wires_df = self.load_data()
        
        print("Exécution de la détection...")
        alerted_accounts_df, flagged_transactions_df = self.run_detection(entity_df, transactions_df, wires_df)
        
        print("Sauvegarde des résultats...")
        self.save_results(alerted_accounts_df, flagged_transactions_df)
        
        print("Détection terminée.")
        if not alerted_accounts_df.empty:
            print(f"{len(alerted_accounts_df)} comptes alertés.")
            print(f"{len(flagged_transactions_df)} transactions marquées.")
            
        return alerted_accounts_df, flagged_transactions_df


if __name__ == "__main__":
    # Exécuter la détection avec la configuration par défaut
    runner = DetectionRunner()
    runner.run() 