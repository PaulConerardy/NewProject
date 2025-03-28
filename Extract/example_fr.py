from oracle_data_extractor import OracleDataExtractor

def main():
    # Initialisation de l'extracteur
    extractor = OracleDataExtractor(
        yaml_file="queries.yaml",
        connection_name="BD_NUM_2"  # Remplacez par votre nom de connexion réel
    )

    # IDs clients d'exemple (à remplacer par les IDs réels)
    client_ids = list(range(1, 5001))  # Exemple: 5000 IDs

    # Extraction des données avec tous les filtres optionnels
    df = extractor.extract(
        query_name="transactions",
        client_ids=client_ids,
        transaction_types=["SALE", "REFUND"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\nNombre total de lignes récupérées: {len(df)}")
    print("\nPremières lignes des données:")
    print(df.head())

if __name__ == "__main__":
    main() 