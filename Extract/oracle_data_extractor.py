import time
import yaml
import pandas as pd
import itc_utils.flight_service as itcfs


class OracleDataExtractor:
    def __init__(self, yaml_file: str, connection_name: str):
        """
        Initialize with YAML query file and Oracle connection name.

        Args:
            yaml_file (str): Path to the YAML file containing SQL queries
            connection_name (str): Oracle connection name for ServiceFlight
        """
        self.connection_name = connection_name
        self.queries = self._load_queries(yaml_file)
        self.client = itcfs.get_flight_client()

    def _load_queries(self, yaml_file: str) -> dict:
        """
        Load SQL queries from YAML file.

        Args:
            yaml_file (str): Path to the YAML file

        Returns:
            dict: Dictionary containing the queries
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
        Execute a query with dynamic WHERE clauses.
        
        Args:
            query_name (str): Name of the query in the YAML file
            client_ids (list): List of client IDs (will be batched into 1000)
            transaction_types (list, optional): List of transaction types to filter
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Combined results from all batches
        """
        base_sql = self.queries[query_name]
        chunks = [client_ids[i:i+1000] for i in range(0, len(client_ids), 1000)]
        dfs = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            # Build WHERE conditions
            conditions = [f"client_id IN ({', '.join(map(str, chunk))})"]
            
            if transaction_types:
                types = ', '.join(f"'{t}'" for t in transaction_types)
                conditions.append(f"transaction_type IN ({types})")
            
            if start_date and end_date:
                conditions.append(
                    f"transaction_date BETWEEN TO_DATE('{start_date}', 'YYYY-MM-DD') "
                    f"AND TO_DATE('{end_date}', 'YYYY-MM-DD')"
                )
            
            # Append conditions to base SQL
            full_sql = f"{base_sql} AND {' AND '.join(conditions)}"
            
            # Execute and time
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
            
            print(f"Batch {chunk_idx}/{len(chunks)} | Rows: {len(df)} | Time: {elapsed:.2f}s")
            dfs.append(df)
        
        return pd.concat(dfs) if dfs else pd.DataFrame() 