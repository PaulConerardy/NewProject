import pandas as pd
import numpy as np
from rule_based_detection import RuleBasedDetection
from network_analysis import NetworkAnalysis
from alerts import Alerts
import os

def load_data():
    """Charge les données de transaction et d'entité à partir des fichiers CSV."""
    # Définir le répertoire de données
    data_dir = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/data"
    
    # Charger les données de virement
    wires_path = os.path.join(data_dir, "synthetic_wires.csv")
    wires_df = pd.read_csv(wires_path, sep='|', skiprows=1)
    
    # Nettoyer les noms de colonnes et les données
    wires_df.columns = ['party_key', 'wire_date', 'sens', 'amount', 'beneficiary', 
                        'originator', 'beneficiary_country', 'originator_country', 
                        'sign', 'transaction_type', 'transaction_type_desc', 'account_type_desc']
    
    # Convertir au format approprié pour le système de détection
    wire_data = pd.DataFrame({
        'wire_id': range(1, len(wires_df) + 1),
        'sender_id': wires_df['party_key'],  # Utilisation de party_key comme ID d'entité
        'destination_id': wires_df['party_key'],  # Ceci est simplifié - dans des données réelles, ces valeurs seraient différentes
        'amount': wires_df['amount'].astype(float),
        'date': pd.to_datetime(wires_df['wire_date'], format='%d%b%Y', errors='coerce'),
        'sender_country': wires_df['originator_country'],
        'destination_country': wires_df['beneficiary_country']
    })
    
    # Créer un dataframe d'entité simple à partir des données de virement
    unique_entities = pd.concat([
        wires_df[['party_key', 'originator']].rename(columns={'originator': 'entity_name'}),
        wires_df[['party_key', 'beneficiary']].rename(columns={'beneficiary': 'entity_name'})
    ]).drop_duplicates('party_key')
    
    entity_data = pd.DataFrame({
        'entity_id': unique_entities['party_key'],
        'entity_name': unique_entities['entity_name'],
        'entity_type': wires_df['account_type_desc'].map({
            'Particulier': 'individual',
            'Entreprise': 'business',
            'Association': 'organization',
            'Trust': 'trust',
            'Joint': 'joint'
        }).fillna('unknown'),
        'country': wires_df['originator_country']  # Simplifié - utilisation du pays d'origine
    }).drop_duplicates('entity_id')
    
    # Pour cet exemple, nous allons créer un dataframe de transaction simple à partir des données de virement
    transaction_data = pd.DataFrame({
        'transaction_id': range(1, len(wires_df) + 1),
        'sender_id': wires_df['party_key'],
        'destination_id': wires_df['party_key'].shift(-1).fillna(wires_df['party_key']),  # Simplifié
        'amount': wires_df['amount'].astype(float),
        'date': pd.to_datetime(wires_df['wire_date'], format='%d%b%Y', errors='coerce'),
        'type': wires_df['transaction_type_desc'].map({
            'Transfert Internet': 'email_transfer',
            'Depot Especes': 'cash_deposit'
        }).fillna('transfer'),
        'location': 'Location ' + wires_df['originator_country']
    })
    
    return entity_data, transaction_data, wire_data

def main():
    """Fonction principale pour exécuter le système de détection AML."""
    print("Chargement des données...")
    entity_data, transaction_data, wire_data = load_data()
    
    print(f"Chargé {len(entity_data)} entités, {len(transaction_data)} transactions, {len(wire_data)} virements")
    
    # Initialiser le système d'alertes
    alerts_system = Alerts()
    
    # Traiter chaque entité et générer des alertes
    print("Traitement des entités...")
    alert_results = []
    
    # Traiter un échantillon d'entités (ajuster le nombre selon les besoins)
    sample_size = min(100, len(entity_data))
    for i, (_, entity) in enumerate(entity_data.head(sample_size).iterrows()):
        if i % 10 == 0:
            print(f"Traitement de l'entité {i+1}/{sample_size}...")
        
        # Convertir la Series d'entité en dict pour le traitement
        entity_dict = entity.to_dict()
        
        # Générer un score pour cette entité
        alert = alerts_system.generate_entity_score(
            entity_dict, entity_data, transaction_data, wire_data
        )
        
        # Stocker les résultats
        alert_results.append({
            'entity_id': entity_dict['entity_id'],
            'entity_name': entity_dict['entity_name'],
            'entity_type': entity_dict['entity_type'],
            'total_score': alert['total_score'],
            'rule_score': alert['rule_score'],
            'network_score': alert['network_score'],
            'is_suspicious': alert['is_suspicious']
        })
    
    # Convertir les résultats en DataFrame
    alerts_df = pd.DataFrame(alert_results)
    
    # Afficher les entités suspectes
    suspicious_entities = alerts_df[alerts_df['is_suspicious']]
    print(f"\nTrouvé {len(suspicious_entities)} entités suspectes:")
    if len(suspicious_entities) > 0:
        print(suspicious_entities[['entity_id', 'entity_name', 'entity_type', 'total_score']])
    
    # Enregistrer les résultats dans un CSV
    output_path = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/alertes_aml.csv"
    alerts_df.to_csv(output_path, index=False)
    print(f"\nRésultats enregistrés dans {output_path}")

if __name__ == "__main__":
    main()