from transaction_generator import TransactionGenerator
from feature_engineer import TransactionFeatureEngineer
import pandas as pd

def main():
    # Generate sample data
    generator = TransactionGenerator(n_customers=1000, n_transactions=10000)
    df = generator.generate_dataset()
    
    # Apply feature engineering
    engineer = TransactionFeatureEngineer()
    df_featured = engineer.engineer_features(df)
    
    # Save to CSV
    output_path = '/Users/paulconerardy/Documents/Trae/Anomaly v2/data/featured_transactions.csv'
    df_featured.to_csv(output_path, index=False)
    print(f"Generated {len(df_featured)} transactions with {len(df_featured.columns)} features")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()