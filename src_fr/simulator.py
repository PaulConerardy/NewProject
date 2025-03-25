import os
import pandas as pd
import yaml
import numpy as np
import json
import itertools
from detection_runner import DetectionRunner
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from visualization import Visualizer
import random
from tqdm import tqdm


class Simulator:
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le Simulateur pour l'optimisation des paramètres.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        # Charger la configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.config_path = config_path
        self.param_grid = self.config['simulation']['param_grid']
        self.evaluation_metric = self.config['simulation']['evaluation_metric']
        
        # Créer le répertoire des résultats
        os.makedirs('simulation_results', exist_ok=True)
        
        # Initialiser le visualiseur
        self.visualizer = Visualizer(output_dir='simulation_results/visualizations')
    
    def _create_param_combinations(self, sampling_method='random', max_combinations=20):
        """
        Créer des combinaisons de paramètres pour la recherche sur grille avec optimisation.
        
        Args:
            sampling_method (str): Méthode à utiliser pour l'échantillonnage des combinaisons.
                                  Options: 'full' (toutes les combinaisons), 'random' (échantillonnage aléatoire)
            max_combinations (int): Nombre maximum de combinaisons à générer pour l'échantillonnage aléatoire
        
        Returns:
            list: Liste des dictionnaires de paramètres.
        """
        param_combinations = []
        
        # Extraire les paramètres des règles
        rule_combinations = {}
        if 'rules' in self.param_grid:
            for rule, params in self.param_grid['rules'].items():
                # Pour chaque règle, créer des combinaisons de seuils et de scores
                rule_combinations[rule] = []
                for threshold_set, score_set in zip(params['thresholds'], params['scores']):
                    rule_combinations[rule].append({
                        'thresholds': threshold_set,
                        'scores': score_set
                    })
        
        # Obtenir les valeurs de bonus pour les drapeaux suspects antérieurs
        prior_flag_boosts = self.param_grid.get('prior_suspicious_flag_boost', 
                                              [self.config.get('prior_suspicious_flag_boost', 20)])
        
        # Créer les combinaisons de paramètres de règles
        rule_names = list(rule_combinations.keys())
        rule_values = [rule_combinations[rule] for rule in rule_names]
        
        if sampling_method == 'full':
            # Recherche sur grille complète - toutes les combinaisons (peut être très lent)
            rule_param_combinations = list(itertools.product(*rule_values))
        elif sampling_method == 'random':
            # Échantillonnage aléatoire des combinaisons
            all_rule_param_combinations = list(itertools.product(*rule_values))
            total_combinations = len(all_rule_param_combinations) * len(prior_flag_boosts)
            
            print(f"Nombre total de combinaisons possibles : {total_combinations}")
            print(f"Utilisation de l'échantillonnage aléatoire pour réduire à max {max_combinations} combinaisons")
            
            # Si le nombre total de combinaisons est inférieur au maximum, les utiliser toutes
            if total_combinations <= max_combinations:
                rule_param_combinations = all_rule_param_combinations
            else:
                # Échantillonner un sous-ensemble de combinaisons de paramètres de règles
                rule_param_combinations = random.sample(all_rule_param_combinations, 
                                                       min(max_combinations // len(prior_flag_boosts), 
                                                           len(all_rule_param_combinations)))
        else:
            # Par défaut, optimisation un facteur à la fois
            # Commencer avec la configuration de base (première option pour chaque paramètre)
            base_combo = tuple(values[0] for values in rule_values)
            rule_param_combinations = [base_combo]
            
            # Puis faire varier un facteur à la fois
            for i, rule_options in enumerate(rule_values):
                if len(rule_options) > 1:
                    for option in rule_options[1:]:
                        # Créer une nouvelle combinaison en remplaçant un seul facteur
                        new_combo = list(base_combo)
                        new_combo[i] = option
                        rule_param_combinations.append(tuple(new_combo))
        
        # Créer les combinaisons finales de paramètres avec le seuil d'alerte fixe de la configuration
        for prior_flag_boost in prior_flag_boosts:
            for rule_combo in rule_param_combinations:
                param_dict = {
                    'alert_threshold': self.config['alert_threshold'],  # Seuil d'alerte fixe de la configuration
                    'prior_suspicious_flag_boost': prior_flag_boost,
                    'rules': {}
                }
                
                # Définir les paramètres des règles
                for i, rule in enumerate(rule_names):
                    param_dict['rules'][rule] = rule_combo[i]
                
                param_combinations.append(param_dict)
        
        return param_combinations
    
    def _update_config(self, params):
        """
        Mettre à jour la configuration avec de nouveaux paramètres.
        
        Args:
            params (dict): Nouveaux paramètres.
            
        Returns:
            str: Chemin vers le fichier de configuration mis à jour.
        """
        # Créer une copie de la configuration originale
        updated_config = self.config.copy()
        
        # Mettre à jour le seuil d'alerte (doit être fixe, mais inclus pour la compatibilité)
        if 'alert_threshold' in params:
            updated_config['alert_threshold'] = params['alert_threshold']
        
        # Mettre à jour le bonus pour les drapeaux suspects antérieurs
        if 'prior_suspicious_flag_boost' in params:
            updated_config['prior_suspicious_flag_boost'] = params['prior_suspicious_flag_boost']
        
        # Mettre à jour les paramètres des règles
        if 'rules' in params:
            for rule, rule_params in params['rules'].items():
                updated_config['rules'][rule] = {
                    'thresholds': rule_params['thresholds'],
                    'scores': rule_params['scores']
                }
        
        # Écrire la configuration mise à jour dans un fichier temporaire
        temp_config_path = 'temp_config.yaml'
        with open(temp_config_path, 'w') as file:
            yaml.dump(updated_config, file)
        
        return temp_config_path
    
    def _evaluate_performance(self, alerted_accounts_df, ground_truth_df=None):
        """
        Évaluer la performance de la détection.
        
        Args:
            alerted_accounts_df (DataFrame): Comptes alertés.
            ground_truth_df (DataFrame, optional): Données de vérité terrain avec les drapeaux suspects réels.
            
        Returns:
            dict: Métriques de performance.
        """
        # Si nous n'avons pas de vérité terrain, utiliser prior_suspicious_flag
        if ground_truth_df is None:
            if 'prior_suspicious_flag' in alerted_accounts_df.columns:
                y_true = alerted_accounts_df['prior_suspicious_flag'].values
                y_pred = np.ones(len(y_true))
                
                # Calculer les métriques
                metrics = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'num_alerts': len(alerted_accounts_df),
                    'true_positives': np.sum(y_true == 1),
                    'false_positives': np.sum(y_true == 0)
                }
                
                return metrics
            else:
                # Sans vérité terrain, retourner uniquement le nombre d'alertes
                return {
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                    'num_alerts': len(alerted_accounts_df)
                }
        else:
            # Utiliser les données de vérité terrain
            # Fusionner les comptes alertés avec la vérité terrain
            merged = pd.merge(
                alerted_accounts_df[['party_key']], 
                ground_truth_df[['party_key', 'is_suspicious']], 
                on='party_key', 
                how='left'
            )
            merged['is_suspicious'] = merged['is_suspicious'].fillna(0)
            
            y_true = merged['is_suspicious'].values
            y_pred = np.ones(len(y_true))
            
            # Calculer les métriques
            metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0) if ground_truth_df['is_suspicious'].sum() > 0 else 0,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'num_alerts': len(alerted_accounts_df),
                'true_positives': np.sum(y_true == 1),
                'false_positives': np.sum(y_true == 0)
            }
            
            return metrics
    
    def run_simulation(self, ground_truth_df=None, sampling_method='random', max_combinations=20):
        """
        Exécuter la simulation pour trouver les paramètres optimaux.
        
        Args:
            ground_truth_df (DataFrame, optional): Données de vérité terrain avec les drapeaux suspects réels.
            sampling_method (str): Méthode pour l'échantillonnage des combinaisons ('full', 'random').
            max_combinations (int): Nombre maximum de combinaisons à tester.
            
        Returns:
            tuple: (best_params, best_metrics)
        """
        print("Exécution de la simulation pour optimiser les paramètres...")
        print(f"Le seuil d'alerte est fixé à {self.config['alert_threshold']}")
        
        # Créer les combinaisons de paramètres
        param_combinations = self._create_param_combinations(
            sampling_method=sampling_method,
            max_combinations=max_combinations
        )
        num_combinations = len(param_combinations)
        print(f"Test de {num_combinations} combinaisons de paramètres...")
        
        # Stocker les résultats
        results = []
        
        # Exécuter la détection avec chaque combinaison de paramètres
        for i, params in enumerate(tqdm(param_combinations, desc="Test des combinaisons"), 1):
            # Mettre à jour la configuration avec les nouveaux paramètres
            temp_config_path = self._update_config(params)
            
            # Exécuter la détection avec la configuration mise à jour
            runner = DetectionRunner(temp_config_path)
            alerted_accounts_df, _ = runner.run()
            
            # Évaluer la performance
            metrics = self._evaluate_performance(alerted_accounts_df, ground_truth_df)
            
            # Ajouter les paramètres aux métriques
            result = {
                'params': params,
                'metrics': metrics
            }
            
            results.append(result)
            
            print(f"  Bonus de drapeau suspect : {params['prior_suspicious_flag_boost']}")
            print(f"  Métriques : {metrics}")
        
        # Trouver la meilleure combinaison de paramètres
        if self.evaluation_metric == 'f1_score':
            best_result = max(results, key=lambda x: x['metrics'].get('f1', -1))
        elif self.evaluation_metric == 'precision':
            best_result = max(results, key=lambda x: x['metrics'].get('precision', -1))
        elif self.evaluation_metric == 'recall':
            best_result = max(results, key=lambda x: x['metrics'].get('recall', -1))
        else:
            # Si aucune métrique valide n'est spécifiée, utiliser le score f1
            best_result = max(results, key=lambda x: x['metrics'].get('f1', -1))
        
        # Sauvegarder tous les résultats
        results_df = pd.DataFrame([
            {
                'prior_suspicious_flag_boost': r['params']['prior_suspicious_flag_boost'],
                'rules': json.dumps(r['params']['rules']),
                'precision': r['metrics'].get('precision', np.nan),
                'recall': r['metrics'].get('recall', np.nan),
                'f1': r['metrics'].get('f1', np.nan),
                'num_alerts': r['metrics'].get('num_alerts', 0),
                'true_positives': r['metrics'].get('true_positives', np.nan),
                'false_positives': r['metrics'].get('false_positives', np.nan)
            }
            for r in results
        ])
        
        results_df.to_csv('simulation_results/param_optimization_results.csv', index=False)
        
        # Générer les visualisations des résultats de simulation
        print("Génération des visualisations des résultats de simulation...")
        self.visualizer.visualize_simulation_results()
        
        # Appliquer les meilleurs paramètres à la configuration principale et sauvegarder
        best_params = best_result['params']
        if best_params:
            print(f"Meilleurs paramètres trouvés :")
            print(f"  - Bonus de drapeau suspect : {best_params['prior_suspicious_flag_boost']}")
            print(f"  - Meilleure {self.evaluation_metric} : {best_result['metrics'].get(self.evaluation_metric.split('_')[0], 'N/A')}")
            
            # Mettre à jour la configuration principale avec les meilleurs paramètres
            updated_config = self.config.copy()
            updated_config['prior_suspicious_flag_boost'] = best_params['prior_suspicious_flag_boost']
            
            if 'rules' in best_params:
                for rule, rule_params in best_params['rules'].items():
                    updated_config['rules'][rule] = {
                        'thresholds': rule_params['thresholds'],
                        'scores': rule_params['scores']
                    }
            
            # Sauvegarder la configuration mise à jour
            with open('config.yaml', 'w') as file:
                yaml.dump(updated_config, file)
            
            print("Configuration principale mise à jour avec les meilleurs paramètres.")
        
        return best_result['params'], best_result['metrics']


if __name__ == "__main__":
    # Exécuter la simulation avec les paramètres optimisés
    simulator = Simulator()
    best_params, best_metrics = simulator.run_simulation(
        sampling_method='random',  # Utiliser l'échantillonnage aléatoire pour l'efficacité
        max_combinations=20        # Limiter à 20 combinaisons
    )
    print(f"Simulation terminée avec la meilleure {simulator.evaluation_metric} : {best_metrics.get(simulator.evaluation_metric.split('_')[0], 'N/A')}") 