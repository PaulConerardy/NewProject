from transaction_generator import TransactionGenerator
import os

def main():
    # Create data directory if it doesn't exist
    data_dir = "/Users/paulconerardy/Documents/Trae/Anomaly v2/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate sample dataset
    generator = TransactionGenerator(n_customers=1000, n_transactions=10000)
    df = generator.generate_dataset(anomaly_ratio=0.05)
    
    # Save to CSV
    output_path = os.path.join(data_dir, "sample_transactions.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions with {df['is_anomaly'].sum()} anomalies")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()