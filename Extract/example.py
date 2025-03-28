from oracle_data_extractor import OracleDataExtractor

def main():
    # Initialize the extractor
    extractor = OracleDataExtractor(
        yaml_file="queries.yaml",
        connection_name="BD_NUM_2"  # Replace with your actual connection name
    )

    # Example client IDs (replace with actual IDs)
    client_ids = list(range(1, 5001))  # Example: 5000 IDs

    # Extract data with all optional filters
    df = extractor.extract(
        query_name="transactions",
        client_ids=client_ids,
        transaction_types=["SALE", "REFUND"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\nTotal rows retrieved: {len(df)}")
    print("\nFirst few rows of the data:")
    print(df.head())

if __name__ == "__main__":
    main() 