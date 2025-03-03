from peer_group_analysis import PeerGroupAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def main():
    # Initialiser l'analyseur
    analyzer = PeerGroupAnalyzer('/Users/paulconerardy/Documents/Trae/ESM3/generated_client_data.csv')
    
    # Prétraiter les données
    print("Prétraitement des données...")
    preprocessed_data = analyzer.preprocess_data()
    
    # Détecter les anomalies en utilisant Isolation Forest
    print("Détection des anomalies en utilisant Isolation Forest...")
    anomalies_if = analyzer.detect_anomalies_by_peer_group(method='isolation_forest', contamination=0.05)
    
    # Obtenir les principales anomalies
    print("\n20 Principales Anomalies:")
    top_anomalies = analyzer.get_top_anomalies(20)
    print(top_anomalies[['party_key', 'party_type', 'peer_group', 'income', 'risk_level', 'anomaly_score']].to_string())
    
    # Analyser les modèles d'anomalie
    print("\nAnalyse des modèles d'anomalie...")
    patterns = analyzer.analyze_anomaly_patterns()
    
    # Imprimer la distribution des anomalies par groupe de pairs
    print("\nDistribution des anomalies par groupe de pairs:")
    print(patterns['anomalies_by_group'])
    
    # Imprimer les schémas potentiels de banque clandestine
    print("\nSchémas potentiels de banque clandestine (flux équilibrés):")
    underground = patterns['potential_underground_banking']
    print(underground[['party_key', 'party_type', 'peer_group', 'flow_balance', 'ACT_PROF_001_VAL', 'ACT_PROF_003_VAL']].head(10).to_string())
    
    # Imprimer le potentiel blanchiment d'argent basé sur le commerce
    print("\nPotentiel blanchiment d'argent basé sur le commerce:")
    trade_ml = patterns['potential_trade_ml']
    print(trade_ml[['party_key', 'party_type', 'peer_group', 'intl_txn_ratio', 'ACT_PROF_RECEIVE_BP_INN_VAL', 'income']].head(10).to_string())
    
    # Visualiser les anomalies
    print("\nVisualisation des anomalies...")
    analyzer.visualize_anomalies()
    
    # Créer des visualisations supplémentaires
    create_additional_visualizations(analyzer.anomaly_scores)
    
    # Sauvegarder les résultats en CSV
    top_anomalies.to_csv('/Users/paulconerardy/Documents/Trae/ESM3/top_anomalies.csv', index=False)
    underground.head(50).to_csv('/Users/paulconerardy/Documents/Trae/ESM3/potential_underground_banking.csv', index=False)
    trade_ml.head(50).to_csv('/Users/paulconerardy/Documents/Trae/ESM3/potential_trade_ml.csv', index=False)
    
    print("\nAnalyse terminée. Résultats sauvegardés dans les fichiers CSV.")

def create_additional_visualizations(data):
    """Créer des visualisations supplémentaires pour l'analyse des anomalies"""
    # Créer une figure avec plusieurs sous-graphiques
    plt.figure(figsize=(16, 12))
    
    # Graphique 1: Distribution des scores d'anomalie
    plt.subplot(2, 2, 1)
    sns.histplot(data=data, x='anomaly_score', hue='is_anomaly', bins=50, kde=True)
    plt.title('Distribution des Scores d\'Anomalie')
    plt.xlabel('Score d\'Anomalie')
    plt.ylabel('Nombre')
    
    # Graphique 2: Scores d'anomalie par niveau de risque
    plt.subplot(2, 2, 2)
    sns.boxplot(data=data, x='risk_level', y='anomaly_score')
    plt.title('Scores d\'Anomalie par Niveau de Risque')
    plt.xlabel('Niveau de Risque')
    plt.ylabel('Score d\'Anomalie')
    
    # Graphique 3: Intensité des transactions vs revenu avec mise en évidence des anomalies
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=data, 
        x='income', 
        y='TXN_INCOME_RATIO',
        hue='is_anomaly',
        size='anomaly_score',
        sizes=(20, 200),
        alpha=0.6
    )
    plt.title('Intensité des Transactions vs Revenu')
    plt.xlabel('Revenu')
    plt.ylabel('Ratio Transaction/Revenu')
    plt.yscale('log')
    plt.xscale('log')
    
    # Graphique 4: Transactions entrantes vs sortantes avec mise en évidence des anomalies
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        data=data, 
        x='ACT_PROF_001_VAL', 
        y='ACT_PROF_003_VAL',
        hue='is_anomaly',
        size='anomaly_score',
        sizes=(20, 200),
        alpha=0.6
    )
    plt.title('Transactions Entrantes vs Sortantes')
    plt.xlabel('Valeur des Transactions Entrantes')
    plt.ylabel('Valeur des Transactions Sortantes')
    plt.yscale('log')
    plt.xscale('log')
    
    # Ajouter une ligne diagonale pour référence (entrantes/sortantes égales)
    max_val = max(data['ACT_PROF_001_VAL'].max(), data['ACT_PROF_003_VAL'].max())
    plt.plot([1, max_val], [1, max_val], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/Trae/ESM3/anomaly_analysis.png', dpi=300)
    plt.show()
    
    # Créer une carte de chaleur de corrélation pour les transactions anormales
    plt.figure(figsize=(12, 10))
    anomalies = data[data['is_anomaly']]
    feature_cols = [col for col in data.columns if col.startswith('ACT_PROF_') or 
                    col.startswith('AVG_') or 
                    col in ['IN_OUT_RATIO', 'TXN_INCOME_RATIO']]
    
    corr = anomalies[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Carte de Chaleur de Corrélation des Caractéristiques de Transaction pour les Anomalies')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/Trae/ESM3/anomaly_correlation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()