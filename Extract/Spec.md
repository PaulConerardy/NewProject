### Specification: Oracle Data Extractor

#### **Overview**
A Python class `OracleDataExtractor` that extracts data from an Oracle database using IBM's `ServiceFlight` package. Key features:
- Reads SQL queries from a YAML file.
- Dynamically splits large `client_id` lists into batches of 1000.
- Supports additional filters: `transaction_types`, `start_date`, and `end_date`.
- Measures and logs the time for each batch request.

---

#### **Requirements**
1. **Simplicity**: Single class, minimal code.
2. **YAML Query Storage**: SQL queries stored in a YAML file under a `queries` key.
3. **Batch Processing**: Split `client_id` lists into chunks of 1000.
4. **Dynamic WHERE Clauses**: Construct conditions for:
   - `client_id` chunks.
   - `transaction_types` (if provided).
   - Date range (if provided).
5. **Diagnostics**: Log time taken for each batch.

---

#### **Class Design**

```python
import time
import yaml
import pandas as pd
import itc_utils.flight_service as itcfs

class OracleDataExtractor:
    def __init__(self, yaml_file: str, connection_name: str):
        """
        Initialize with YAML query file and Oracle connection name.
        """
        self.connection_name = connection_name
        self.queries = self._load_queries(yaml_file)
        self.client = itcfs.get_flight_client()

    def _load_queries(self, yaml_file: str) -> dict:
        """Load SQL queries from YAML."""
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
            query_name: Name of the query in the YAML file.
            client_ids: List of client IDs (batched into 1000).
            transaction_types: Optional list of transaction types.
            start_date/end_date: Optional date range (format: 'YYYY-MM-DD').
        """
        base_sql = self.queries[query_name]
        chunks = [client_ids[i:i+1000] for i in range(0, len(client_ids), 1000)]
        dfs = []

        for chunk in chunks:
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
            
            print(f"Batch {len(df)} rows | Time: {elapsed:.2f}s")
            dfs.append(df)
        
        return pd.concat(dfs)
```

---

#### **YAML File Example**
```yaml
queries:
  transactions: |
    SELECT * FROM "MYSCHEMA"."MYTABLE" WHERE 1=1
```

---

#### **Usage Example**
```python
extractor = OracleDataExtractor("queries.yaml", "BD_NUM_2")
df = extractor.extract(
    query_name="transactions",
    client_ids=[1, 2, 3, ..., 5000],
    transaction_types=["SALE", "REFUND"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

---

#### **Dependencies**
- `PyYAML`: For parsing the YAML file.
- `pandas`: For DataFrame concatenation.
- `itc_utils.flight_service`: IBM ServiceFlight package.

---

#### **Key Notes**
1. **Batching**: Splits `client_ids` into chunks (e.g., 5000 IDs → 5 batches of 1000).
2. **SQL Safety**: Assumes trusted inputs (no injection handling for simplicity).
3. **Date Format**: Uses Oracle’s `TO_DATE` with `'YYYY-MM-DD'`.
4. **Logging**: Prints batch size and time for diagnostics.