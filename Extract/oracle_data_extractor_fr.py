import time
import yaml
import pandas as pd
import itc_utils.flight_service as itcfs


class OracleDataExtractor:
    def __init__(self, yaml_file: str, connection_name: str):
        """
        Initialisation avec le fichier YAML de requêtes et le nom de connexion Oracle.

        Args:
            yaml_file (str): Chemin vers le fichier YAML contenant les requêtes SQL
            connection_name (str): Nom de la connexion Oracle pour ServiceFlight
        """
        self.connection_name = connection_name
        self.queries = self._load_queries(yaml_file)
        self.client = itcfs.get_flight_client()

    def _load_queries(self, yaml_file: str) -> dict:
        """
        Charger les requêtes SQL depuis le fichier YAML.

        Args:
            yaml_file (str): Chemin vers le fichier YAML

        Returns:
            dict: Dictionnaire contenant les requêtes
        """
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)['queries']

    def extract(
        self,
        query_name: str,
        client_ids: list,
        transaction_types: list = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Exécuter une requête avec des clauses WHERE dynamiques.
        
        Args:
            query_name (str): Nom de la requête dans le fichier YAML
            client_ids (list): Liste des IDs clients (sera divisée en lots de 1000)
            transaction_types (list, optional): Liste des types de transactions à filtrer
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'

        Returns:
            pd.DataFrame: Résultats combinés de tous les lots
        """
        base_sql = self.queries[query_name]
        chunks = [client_ids[i:i+1000] for i in range(0, len(client_ids), 1000)]
        dfs = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            # Construction des conditions WHERE
            conditions = [f"client_id IN ({', '.join(map(str, chunk))})"]
            
            if transaction_types:
                types = ', '.join(f"'{t}'" for t in transaction_types)
                conditions.append(f"transaction_type IN ({types})")
            
            if start_date and end_date:
                conditions.append(
                    f"transaction_date BETWEEN TO_DATE('{start_date}', 'YYYY-MM-DD') "
                    f"AND TO_DATE('{end_date}', 'YYYY-MM-DD')"
                )
            
            # Ajout des conditions à la requête SQL de base
            full_sql = f"{base_sql} AND {' AND '.join(conditions)}"
            
            # Exécution et chronométrage
            start_time = time.time()
            flight_info = itcfs.get_flight_info(
                self.client,
                nb_data_request={
                    'connection_name': self.connection_name,
                    'interaction_properties': {'select_statement': full_sql}
                }
            )
            df = itcfs.read_pandas_and_concat(self.client, flight_info, timeout=240)
            elapsed = time.time() - start_time
            
            print(f"Lot {chunk_idx}/{len(chunks)} | Lignes: {len(df)} | Temps: {elapsed:.2f}s")
            dfs.append(df)
        
        return pd.concat(dfs) if dfs else pd.DataFrame() 