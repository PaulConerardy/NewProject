import serviceflight as sf
from typing import Optional, Dict, Any, List
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import yaml

class DataExtractor:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialise l'extracteur de données pour la connexion à Oracle via ServiceFlight.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration contenant les paramètres de connexion.
        """
        # Initialiser Spark
        self.spark = SparkSession.builder \
            .appName("AML Data Extractor") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
        # Charger la configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Extraire les paramètres de connexion
        self.db_config = self.config.get('database', {})
        self.connection = None
        
        # Liste des entités à traiter
        self.target_entities = None
        
        # Initialiser la connexion
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialise la connexion à la base de données Oracle via ServiceFlight."""
        try:
            self.connection = sf.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 1521),
                service_name=self.db_config.get('service_name'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password')
            )
        except Exception as e:
            print(f"Erreur lors de la connexion à la base de données: {str(e)}")
            raise
            
    def load_target_entities(self, parquet_path: str) -> None:
        """
        Charge la liste des entités cibles depuis un fichier parquet.
        
        Args:
            parquet_path (str): Chemin vers le fichier parquet contenant la liste des entités.
        """
        try:
            # Lire le fichier parquet avec Spark
            entities_df = self.spark.read.parquet(parquet_path)
            
            # Vérifier que la colonne party_key existe
            if 'party_key' not in entities_df.columns:
                raise ValueError("Le fichier parquet doit contenir une colonne 'party_key'")
            
            # Extraire la liste des party_key uniques
            self.target_entities = [row.party_key for row in entities_df.select('party_key').distinct().collect()]
            
            print(f"Chargé {len(self.target_entities)} entités cibles depuis {parquet_path}")
            
        except Exception as e:
            print(f"Erreur lors du chargement des entités cibles: {str(e)}")
            raise
            
    def _format_entity_list_for_query(self) -> str:
        """
        Formate la liste des entités pour l'utiliser dans une clause IN de SQL.
        
        Returns:
            str: Liste formatée pour SQL, ex: "'entity1', 'entity2', 'entity3'"
        """
        if not self.target_entities:
            return ""
            
        # Échapper les valeurs et les entourer de guillemets
        formatted_values = [f"'{str(entity)}'" for entity in self.target_entities]
        return ", ".join(formatted_values)
            
    def _execute_query(self, query: str) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne les résultats.
        
        Args:
            query (str): Requête SQL à exécuter.
            
        Returns:
            pd.DataFrame: Résultats de la requête.
        """
        try:
            if not self.connection:
                self._initialize_connection()
                
            # Exécuter la requête via ServiceFlight
            result = sf.read_sql(query, self.connection)
            return result
            
        except Exception as e:
            print(f"Erreur lors de l'exécution de la requête: {str(e)}")
            raise
            
    def _convert_to_spark_df(self, pandas_df: pd.DataFrame):
        """
        Convertit un DataFrame pandas en DataFrame Spark.
        
        Args:
            pandas_df (pd.DataFrame): DataFrame pandas à convertir.
            
        Returns:
            pyspark.sql.DataFrame: DataFrame Spark correspondant.
        """
        if pandas_df.empty:
            return None
            
        return self.spark.createDataFrame(pandas_df)
        
    def get_entities(self) -> Optional[Any]:
        """
        Récupère les données des entités depuis la base de données.
        Si une liste d'entités cibles est définie, seules ces entités seront récupérées.
        
        Returns:
            pyspark.sql.DataFrame: DataFrame Spark contenant les données des entités.
        """
        base_query = self.db_config.get('queries', {}).get('entities', """
            SELECT 
                party_key,
                account_key,
                account_type_desc,
                prior_suspicious_flag
            FROM aml_entities
        """)
        
        # Ajouter la clause WHERE si des entités cibles sont définies
        if self.target_entities:
            entity_list = self._format_entity_list_for_query()
            query = f"{base_query.strip()} WHERE party_key IN ({entity_list})"
        else:
            query = base_query
        
        result_df = self._execute_query(query)
        return self._convert_to_spark_df(result_df)
        
    def get_transactions(self) -> Optional[Any]:
        """
        Récupère les données des transactions depuis la base de données.
        Si une liste d'entités cibles est définie, seules leurs transactions seront récupérées.
        
        Returns:
            pyspark.sql.DataFrame: DataFrame Spark contenant les données des transactions.
        """
        base_query = self.db_config.get('queries', {}).get('transactions', """
            SELECT 
                party_key,
                trx_date,
                amount,
                transaction_type_desc,
                sign,
                branch,
                account_type_desc
            FROM aml_transactions
        """)
        
        # Ajouter la clause WHERE si des entités cibles sont définies
        if self.target_entities:
            entity_list = self._format_entity_list_for_query()
            query = f"{base_query.strip()} WHERE party_key IN ({entity_list})"
        else:
            query = base_query
        
        result_df = self._execute_query(query)
        return self._convert_to_spark_df(result_df)
        
    def get_wires(self) -> Optional[Any]:
        """
        Récupère les données des virements depuis la base de données.
        Si une liste d'entités cibles est définie, seuls leurs virements seront récupérés.
        
        Returns:
            pyspark.sql.DataFrame: DataFrame Spark contenant les données des virements.
        """
        base_query = self.db_config.get('queries', {}).get('wires', """
            SELECT 
                party_key,
                wire_date,
                amount,
                sign,
                originator,
                originator_country,
                beneficiary,
                beneficiary_country
            FROM aml_wires
        """)
        
        # Ajouter la clause WHERE si des entités cibles sont définies
        if self.target_entities:
            entity_list = self._format_entity_list_for_query()
            query = f"{base_query.strip()} WHERE party_key IN ({entity_list})"
        else:
            query = base_query
        
        result_df = self._execute_query(query)
        return self._convert_to_spark_df(result_df)
        
    def close(self):
        """Ferme la connexion à la base de données."""
        if self.connection:
            self.connection.close()
            self.connection = None 