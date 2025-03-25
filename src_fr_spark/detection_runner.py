import os
import yaml
import json
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from rule_based_detection import RuleBasedDetection
from visualization import Visualizer

class DetectionRunner:
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le Runner de Détection avec la configuration.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration.
        """
        # Initialiser Spark
        self.spark = SparkSession.builder \
            .appName("AML Detection Runner") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
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
            entity_df = self.spark.read.csv(self.input_paths['entities'], header=True, inferSchema=True)
            # Ajouter la colonne prior_suspicious_flag si elle n'existe pas
            if 'prior_suspicious_flag' not in entity_df.columns:
                entity_df = entity_df.withColumn('prior_suspicious_flag', F.lit(0))
        
        # Charger les données des transactions
        if 'transactions' in self.input_paths and os.path.exists(self.input_paths['transactions']):
            transactions_df = self.spark.read.csv(self.input_paths['transactions'], header=True, inferSchema=True)
        
        # Charger les données des virements
        if 'wires' in self.input_paths and os.path.exists(self.input_paths['wires']):
            wires_df = self.spark.read.csv(self.input_paths['wires'], header=True, inferSchema=True)
        
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
        # Obtenir les IDs uniques des entités
        entity_ids = []
        
        if entity_df is not None and entity_df.count() > 0:
            entity_ids = [row[0] for row in entity_df.select('party_key').distinct().collect()]
        elif transactions_df is not None and transactions_df.count() > 0:
            entity_ids = [row[0] for row in transactions_df.select('party_key').distinct().collect()]
        elif wires_df is not None and wires_df.count() > 0:
            entity_ids = [row[0] for row in wires_df.select('party_key').distinct().collect()]
        else:
            # Créer un DataFrame vide avec un schéma approprié pour les comptes alertés
            alerted_schema = StructType([
                StructField("party_key", StringType(), True),
                StructField("account_key", StringType(), True),
                StructField("total_score", DoubleType(), True),
                StructField("triggered_rules", StringType(), True),
                StructField("prior_suspicious_flag", IntegerType(), True)
            ])
            # Créer un DataFrame vide avec un schéma approprié pour les transactions marquées
            transactions_schema = StructType([])
            
            return self.spark.createDataFrame([], alerted_schema), self.spark.createDataFrame([], transactions_schema)
        
        # Préparer les structures pour les DataFrames de résultats
        alerted_accounts = []
        flagged_transactions_list = []
        
        # Traiter chaque entité
        for entity_id in entity_ids:
            # Calculer le score pour l'entité
            score, triggered_rules = self.detector.calculate_score(entity_id, transactions_df, wires_df, entity_df)
            
            # Si le score est supérieur au seuil, alerter le compte
            if score >= self.alert_threshold:
                # Obtenir les détails de l'entité
                entity_info = {}
                if entity_df is not None and entity_df.count() > 0:
                    entity_row = entity_df.filter(F.col("party_key") == entity_id).first()
                    if entity_row:
                        entity_info = entity_row.asDict()
                
                # Ajouter aux comptes alertés
                account_record = {
                    'party_key': entity_id,
                    'account_key': entity_info.get('account_key') if entity_info else None,
                    'total_score': score,
                    'triggered_rules': json.dumps(triggered_rules),
                    'prior_suspicious_flag': entity_info.get('prior_suspicious_flag', 0) if entity_info else 0
                }
                alerted_accounts.append(account_record)
                
                # Marquer toutes les transactions pour cette entité
                if transactions_df is not None and transactions_df.count() > 0:
                    entity_txns = transactions_df.filter(F.col("party_key") == entity_id)
                    if entity_txns.count() > 0:
                        # Ajouter les informations de marquage
                        entity_txns = entity_txns.withColumn('is_flagged', F.lit(1)) \
                                               .withColumn('total_score', F.lit(score))
                        
                        # Déterminer quelles règles ont contribué à chaque transaction
                        for rule, rule_score in triggered_rules.items():
                            entity_txns = entity_txns.withColumn(f'rule_{rule}', F.lit(rule_score))
                        
                        flagged_transactions_list.append(entity_txns)
                
                # Marquer tous les virements pour cette entité
                if wires_df is not None and wires_df.count() > 0:
                    entity_wires = wires_df.filter(F.col("party_key") == entity_id)
                    if entity_wires.count() > 0:
                        # Ajouter les informations de marquage
                        entity_wires = entity_wires.withColumn('is_flagged', F.lit(1)) \
                                                  .withColumn('total_score', F.lit(score))
                        
                        # Déterminer quelles règles ont contribué à chaque virement
                        for rule, rule_score in triggered_rules.items():
                            entity_wires = entity_wires.withColumn(f'rule_{rule}', F.lit(rule_score))
                        
                        flagged_transactions_list.append(entity_wires)
        
        # Créer les DataFrames finaux
        alerted_accounts_schema = StructType([
            StructField("party_key", StringType(), True),
            StructField("account_key", StringType(), True),
            StructField("total_score", DoubleType(), True),
            StructField("triggered_rules", StringType(), True),
            StructField("prior_suspicious_flag", IntegerType(), True)
        ])
        
        alerted_accounts_df = self.spark.createDataFrame(alerted_accounts, alerted_accounts_schema) if alerted_accounts else self.spark.createDataFrame([], alerted_accounts_schema)
        
        # Combiner les transactions marquées
        if flagged_transactions_list:
            # Union all dataframes in the list
            flagged_transactions_df = flagged_transactions_list[0]
            for df in flagged_transactions_list[1:]:
                flagged_transactions_df = flagged_transactions_df.unionByName(df, allowMissingColumns=True)
        else:
            # Créer un DataFrame vide avec la structure appropriée
            flagged_transactions_df = self.spark.createDataFrame([], transactions_df.schema if transactions_df else StructType([]))
        
        return alerted_accounts_df, flagged_transactions_df
    
    def save_results(self, alerted_accounts_df, flagged_transactions_df):
        """
        Sauvegarder les résultats de détection dans les fichiers de sortie.
        
        Args:
            alerted_accounts_df (DataFrame): Données des comptes alertés.
            flagged_transactions_df (DataFrame): Données des transactions marquées.
        """
        if alerted_accounts_df.count() > 0:
            alerted_accounts_df.write.csv(self.output_paths['alerted_accounts'], header=True, mode="overwrite")
            print(f"Comptes alertés sauvegardés dans {self.output_paths['alerted_accounts']}")
        else:
            print("Aucun compte n'a été alerté.")
        
        if flagged_transactions_df.count() > 0:
            flagged_transactions_df.write.csv(self.output_paths['flagged_transactions'], header=True, mode="overwrite")
            print(f"Transactions marquées sauvegardées dans {self.output_paths['flagged_transactions']}")
        else:
            print("Aucune transaction n'a été marquée.")
        
        # Générer les visualisations
        if alerted_accounts_df.count() > 0:
            print("Génération des visualisations...")
            # Convertir les DataFrame Spark en DataFrame pandas pour la visualisation
            alerted_accounts_pd = alerted_accounts_df.toPandas()
            flagged_transactions_pd = flagged_transactions_df.toPandas()
            
            self.visualizer.visualize_detection_results(alerted_accounts_pd, flagged_transactions_pd)
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
        if alerted_accounts_df.count() > 0:
            print(f"{alerted_accounts_df.count()} comptes alertés.")
            print(f"{flagged_transactions_df.count()} transactions marquées.")
            
        return alerted_accounts_df, flagged_transactions_df


if __name__ == "__main__":
    # Exécuter la détection avec la configuration par défaut
    runner = DetectionRunner()
    runner.run() 