import pandas as pd
import numpy as np
import os
from rule_based_detection import RuleBasedDetection

def load_data():
    """Charge les données synthétiques à partir des fichiers CSV."""
    # Définir le répertoire de données
    data_dir = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/data"
    
    # Charger les données d'entité
    entity_path = os.path.join(data_dir, "synthetic_entity.csv")
    entity_df = pd.read_csv(entity_path, sep='|', skiprows=1)
    
    # Charger les données de transaction
    trx_path = os.path.join(data_dir, "synthetic_trx.csv")
    trx_df = pd.read_csv(trx_path, sep='|', skiprows=1)
    
    # Charger les données de virement
    wires_path = os.path.join(data_dir, "synthetic_wires.csv")
    wires_df = pd.read_csv(wires_path, sep='|', skiprows=1)
    
    return entity_df, trx_df, wires_df

def analyze_entities(entity_df, trx_df, wires_df):
    """Analyse les entités pour détecter des activités suspectes."""
    # Initialiser le détecteur basé sur les règles
    rule_detector = RuleBasedDetection()
    
    # Stocker les résultats
    results = []
    
    # Analyser chaque entité unique
    unique_entities = entity_df['party_key'].unique()
    
    print(f"Analyse de {len(unique_entities)} entités...")
    
    for i, entity_id in enumerate(unique_entities):
        if i % 10 == 0:
            print(f"Progression: {i}/{len(unique_entities)} entités analysées")
        
        # Calculer le score basé sur les règles
        rule_score = rule_detector.calculate_score(entity_id, trx_df, wires_df, entity_df)
        
        # Si le score est supérieur à zéro, obtenir les détails des règles déclenchées
        rule_details = {}
        if rule_score > 0:
            rule_details = rule_detector.get_rule_details(entity_id, trx_df, wires_df, entity_df)
        
        # Obtenir les informations de l'entité
        entity_info = entity_df[entity_df['party_key'] == entity_id].iloc[0]
        
        # Déterminer si l'entité est suspecte (score > 15)
        is_suspicious = rule_score >= 15
        
        # Stocker les résultats
        results.append({
            'entity_id': entity_id,
            'entity_name': entity_info['party_name'],
            'entity_type': entity_info['account_type_desc'],
            'rule_score': rule_score,
            'is_suspicious': is_suspicious,
            'triggered_rules': rule_details
        })
    
    return results

def generate_report(results):
    """Génère un rapport détaillé des entités suspectes."""
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame([
        {
            'entity_id': r['entity_id'],
            'entity_name': r['entity_name'],
            'entity_type': r['entity_type'],
            'rule_score': r['rule_score'],
            'is_suspicious': r['is_suspicious'],
            'triggered_rules_count': len(r['triggered_rules'])
        } for r in results
    ])
    
    # Filtrer les entités suspectes
    suspicious_df = results_df[results_df['is_suspicious']]
    
    print("\n=== RAPPORT DE DÉTECTION ===")
    print(f"Total des entités analysées: {len(results_df)}")
    print(f"Entités suspectes détectées: {len(suspicious_df)}")
    
    if len(suspicious_df) > 0:
        print("\nTop 10 des entités les plus suspectes:")
        top_suspicious = suspicious_df.sort_values('rule_score', ascending=False).head(10)
        for _, row in top_suspicious.iterrows():
            print(f"ID: {row['entity_id']}, Nom: {row['entity_name']}, Type: {row['entity_type']}, Score: {row['rule_score']}")
            
            # Afficher les règles déclenchées pour cette entité
            entity_result = next(r for r in results if r['entity_id'] == row['entity_id'])
            rule_descriptions = RuleBasedDetection().get_rule_descriptions()
            
            print("  Règles déclenchées:")
            for rule, score in entity_result['triggered_rules'].items():
                print(f"  - {rule_descriptions[rule]} (Score: {score})")
            print()
    
    # Enregistrer les résultats complets dans un CSV
    output_path = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/new/detection_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Résultats complets enregistrés dans {output_path}")
    
    # Enregistrer les détails des entités suspectes dans un CSV séparé
    if len(suspicious_df) > 0:
        suspicious_output_path = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/new/suspicious_entities.csv"
        suspicious_df.to_csv(suspicious_output_path, index=False)
        print(f"Détails des entités suspectes enregistrés dans {suspicious_output_path}")
    
    return results_df, suspicious_df

def main():
    """Fonction principale pour exécuter le système de détection."""
    print("Chargement des données...")
    entity_df, trx_df, wires_df = load_data()
    
    print(f"Données chargées: {len(entity_df)} entités, {len(trx_df)} transactions, {len(wires_df)} virements")
    
    print("Analyse des entités...")
    results = analyze_entities(entity_df, trx_df, wires_df)
    
    print("Génération du rapport...")
    results_df, suspicious_df = generate_report(results)
    
    print("\nAnalyse terminée.")

if __name__ == "__main__":
    main()