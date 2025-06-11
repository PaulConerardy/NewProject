import os
import pandas as pd
import yaml
import numpy as np
from detection_runner import DetectionRunner
from sklearn.metrics import precision_score, recall_score, f1_score

class SimplifiedSimulator:
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le Simulateur Simplifié avec la configuration.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        # Charger la configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.config_path = config_path
        self.evaluation_metric = self.config['simulation'].get('evaluation_metric', 'f1_score') # Default to f1_score
        
    def _evaluate_performance(self, alerted_accounts_df, ground_truth_df=None):
        """
        Évaluer la performance de la détection.
        
        Args:
            alerted_accounts_df (DataFrame): Comptes alertés.
            ground_truth_df (DataFrame, optional): Données de vérité terrain avec les drapeaux suspects réels.
            
        Returns:
            dict: Métriques de performance.
        """
        # If no ground truth, use prior_suspicious_flag
        if ground_truth_df is None:
            if 'prior_suspicious_flag' in alerted_accounts_df.columns:
                # Assuming entities in alerted_accounts_df are flagged as 1, compare with prior_suspicious_flag
                # This is a simplification; a true ground truth would be ideal.
                y_true = alerted_accounts_df['prior_suspicious_flag'].values
                y_pred = np.ones(len(y_true)) # All alerted accounts are predicted as positive

                # If there are no positive actuals in y_true, precision, recall and f1_score might be undefined.
                # Handle this by checking if sum(y_true) > 0 before calculating recall and f1.
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0) if np.sum(y_true) > 0 else 0
                f1 = f1_score(y_true, y_pred, zero_division=0) if (precision + recall) > 0 else 0
                
                metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'num_alerts': len(alerted_accounts_df),
                    'true_positives': np.sum(y_true == 1),
                    'false_positives': np.sum(y_true == 0)
                }
                
                return metrics
            else:
                # Without ground truth, return only the number of alerts
                return {
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                    'num_alerts': len(alerted_accounts_df)
                }
        else:
            # Use ground truth data
            merged = pd.merge(
                alerted_accounts_df[['party_key']], 
                ground_truth_df[['party_key', 'is_suspicious']], 
                on='party_key', 
                how='left'
            )
            merged['is_suspicious'] = merged['is_suspicious'].fillna(0).astype(int) # Ensure integer type

            # Get all unique party_keys from ground_truth_df
            all_actual_party_keys = ground_truth_df['party_key'].unique()

            # Create a full prediction set for all entities in ground truth
            y_true_full = ground_truth_df['is_suspicious'].values
            y_pred_full = np.zeros(len(all_actual_party_keys))

            # Mark predicted positives for those in alerted_accounts_df
            alerted_party_keys = alerted_accounts_df['party_key'].unique()
            for i, pk in enumerate(all_actual_party_keys):
                if pk in alerted_party_keys:
                    y_pred_full[i] = 1

            # Calculate metrics
            precision = precision_score(y_true_full, y_pred_full, zero_division=0)
            recall = recall_score(y_true_full, y_pred_full, zero_division=0) if np.sum(y_true_full) > 0 else 0
            f1 = f1_score(y_true_full, y_pred_full, zero_division=0) if (precision + recall) > 0 else 0
            
            # Confusion matrix components for additional insights
            tn, fp, fn, tp = confusion_matrix(y_true_full, y_pred_full, labels=[0, 1]).ravel()

            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_alerts': len(alerted_accounts_df),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            return metrics
    
    def run_single_detection(self, ground_truth_df=None):
        """
        Exécuter une détection unique avec la configuration actuelle et évaluer la performance.
        
        Args:
            ground_truth_df (DataFrame, optional): Données de vérité terrain avec les drapeaux suspects réels.
            
        Returns:
            dict: Métriques de performance.
        """
        print("Exécution d'une détection unique avec la configuration actuelle...")
        
        # Exécuter la détection
        runner = DetectionRunner(self.config_path)
        entity_df, transactions_df, wires_df = runner.load_data()
        alerted_accounts_df, flagged_transactions_df = runner.run_detection(entity_df, transactions_df, wires_df)
        
        # Évaluer la performance
        metrics = self._evaluate_performance(alerted_accounts_df, ground_truth_df)
        
        print("Résultats de la détection unique :")
        print(f"  Nombre de comptes alertés : {metrics.get('num_alerts', 0)}")
        print(f"  Précision : {metrics.get('precision', np.nan):.4f}")
        print(f"  Rappel : {metrics.get('recall', np.nan):.4f}")
        print(f"  Score F1 : {metrics.get('f1', np.nan):.4f}")
        
        return metrics

if __name__ == "__main__":
    # Exécuter la détection unique avec la configuration par défaut
    simulator = SimplifiedSimulator()
    simulator.run_single_detection() 