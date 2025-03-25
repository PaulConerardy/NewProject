import os
import yaml
import numpy as np
import json
import itertools
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
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
        # Initialiser Spark
        self.spark = SparkSession.builder \
            .appName("AML Simulator") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
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
                # Collecte les données pour le calcul des métriques
                alerted_data = alerted_accounts_df.select('prior_suspicious_flag').collect()
                y_true = np.array([row['prior_suspicious_flag'] for row in alerted_data])
                y_pred = np.ones(len(y_true))
                
                # Calculer les métriques
                metrics = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'num_alerts': alerted_accounts_df.count(),
                    'true_positives': int(np.sum(y_true == 1)),
                    'false_positives': int(np.sum(y_true == 0))
                }
                
                return metrics
            else:
                # Sans vérité terrain, retourner uniquement le nombre d'alertes
                return {
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                    'num_alerts': alerted_accounts_df.count()
                }
        else:
            # Utiliser les données de vérité terrain
            # Joindre les comptes alertés avec la vérité terrain
            merged = alerted_accounts_df.select('party_key').join(
                ground_truth_df.select('party_key', 'is_suspicious'),
                on='party_key',
                how='left'
            )
            
            # Remplacer les valeurs None par 0
            merged = merged.fillna({'is_suspicious': 0})
            
            # Collecte les données pour le calcul des métriques
            merged_data = merged.collect()
            y_true = np.array([row['is_suspicious'] for row in merged_data])
            y_pred = np.ones(len(y_true))
            
            # Calculer le nombre total de vrais positifs dans ground_truth
            total_positives = ground_truth_df.filter(F.col('is_suspicious') == 1).count()
            
            # Calculer les métriques
            metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0) if total_positives > 0 else 0,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'num_alerts': alerted_accounts_df.count(),
                'true_positives': int(np.sum(y_true == 1)),
                'false_positives': int(np.sum(y_true == 0))
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
            
            try:
                # Exécuter la détection avec la configuration mise à jour
                runner = DetectionRunner(config_path=temp_config_path)
                entity_df, transactions_df, wires_df = runner.load_data()
                alerted_accounts_df, flagged_transactions_df = runner.run_detection(entity_df, transactions_df, wires_df)
                
                # Évaluer la performance
                metrics = self._evaluate_performance(alerted_accounts_df, ground_truth_df)
                
                # Stocker les résultats
                results.append({
                    'params': params,
                    'metrics': metrics
                })
                
                print(f"Combinaison {i}/{num_combinations}: "
                      f"Alertes: {metrics['num_alerts']}, "
                      f"F1: {metrics['f1']:.4f}, "
                      f"Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}")
            
            except Exception as e:
                print(f"Erreur lors du test de la combinaison {i}: {str(e)}")
                # Continuer avec la combinaison suivante
        
        # Trouver les meilleurs paramètres
        if results:
            # Trier les résultats en fonction de la métrique d'évaluation
            if self.evaluation_metric == 'f1_score':
                sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
            elif self.evaluation_metric == 'precision':
                sorted_results = sorted(results, key=lambda x: x['metrics']['precision'], reverse=True)
            elif self.evaluation_metric == 'recall':
                sorted_results = sorted(results, key=lambda x: x['metrics']['recall'], reverse=True)
            else:  # Par défaut, utiliser F1
                sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
            
            # Meilleurs paramètres
            best_result = sorted_results[0]
            best_params = best_result['params']
            best_metrics = best_result['metrics']
            
            print("\nMeilleurs paramètres trouvés:")
            print(f"Seuil d'alerte: {best_params['alert_threshold']}")
            print(f"Bonus pour les drapeaux suspects antérieurs: {best_params['prior_suspicious_flag_boost']}")
            print("Paramètres des règles:")
            for rule, rule_params in best_params['rules'].items():
                print(f"  {rule}:")
                print(f"    Seuils: {rule_params['thresholds']}")
                print(f"    Scores: {rule_params['scores']}")
            
            print("\nMeilleures métriques:")
            print(f"F1: {best_metrics['f1']:.4f}")
            print(f"Precision: {best_metrics['precision']:.4f}")
            print(f"Recall: {best_metrics['recall']:.4f}")
            print(f"Nombre d'alertes: {best_metrics['num_alerts']}")
            if 'true_positives' in best_metrics and 'false_positives' in best_metrics:
                print(f"Vrais positifs: {best_metrics['true_positives']}")
                print(f"Faux positifs: {best_metrics['false_positives']}")
            
            # Sauvegarder les meilleurs paramètres dans un fichier de configuration final
            best_config_path = 'simulation_results/best_config.yaml'
            self._update_config(best_params)
            os.rename('temp_config.yaml', best_config_path)
            print(f"\nMeilleure configuration sauvegardée dans {best_config_path}")
            
            # Sauvegarder tous les résultats
            all_results_path = 'simulation_results/all_results.json'
            with open(all_results_path, 'w') as f:
                # Convertir les np.int64, np.float64, etc. en types intégrés de Python
                results_json = json.dumps([{
                    'params': r['params'],
                    'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                                int(v) if isinstance(v, (np.int32, np.int64)) else v 
                                for k, v in r['metrics'].items()}
                } for r in results], indent=2)
                f.write(results_json)
            print(f"Tous les résultats sauvegardés dans {all_results_path}")
            
            # Visualiser les résultats
            try:
                # Convertir les résultats pour la visualisation
                results_for_viz = [{
                    'param_id': i,
                    'alert_threshold': r['params']['alert_threshold'],
                    'prior_suspicious_flag_boost': r['params']['prior_suspicious_flag_boost'],
                    'f1': r['metrics']['f1'],
                    'precision': r['metrics']['precision'],
                    'recall': r['metrics']['recall'],
                    'num_alerts': r['metrics']['num_alerts']
                } for i, r in enumerate(results)]
                
                # Créer un DataFrame pandas pour la visualisation
                import pandas as pd
                results_df = pd.DataFrame(results_for_viz)
                
                # Générer les visualisations
                self.visualizer.visualize_simulation_results(results_df)
                print(f"Visualisations des résultats sauvegardées dans {self.visualizer.output_dir}")
            except Exception as e:
                print(f"Erreur lors de la génération des visualisations: {str(e)}")
            
            return best_params, best_metrics
        else:
            print("Aucun résultat obtenu. La simulation a échoué.")
            return None, None

    def run(self, sampling_method='random', max_combinations=20):
        """
        Exécuter la simulation complète.
        
        Args:
            sampling_method (str): Méthode pour l'échantillonnage des combinaisons.
            max_combinations (int): Nombre maximum de combinaisons à tester.
            
        Returns:
            tuple: (best_params, best_metrics)
        """
        # Charger les données de vérité terrain si disponibles
        ground_truth_path = 'data/ground_truth.csv'
        ground_truth_df = None
        
        if os.path.exists(ground_truth_path):
            print(f"Chargement des données de vérité terrain depuis {ground_truth_path}")
            ground_truth_df = self.spark.read.csv(ground_truth_path, header=True, inferSchema=True)
        else:
            print("Aucune donnée de vérité terrain trouvée. Utilisation des drapeaux suspects antérieurs pour l'évaluation.")
        
        # Exécuter la simulation
        return self.run_simulation(
            ground_truth_df=ground_truth_df,
            sampling_method=sampling_method,
            max_combinations=max_combinations
        )


if __name__ == "__main__":
    # Exécuter la simulation avec la configuration par défaut
    simulator = Simulator()
    simulator.run(sampling_method='random', max_combinations=20) 