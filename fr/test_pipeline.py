import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ajout de src au chemin
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.transaction_generator import TransactionGenerator
from features.fr.feature_engineer_fr import TransactionFeatureEngineer
from features.fr.network_features import TransactionNetworkAnalyzer
from models.fr.ml_detector import MLDetector

def test_data_generation():
    """
    Test de la génération des données de transaction.
    
    Génère un jeu de données synthétique et effectue des validations
    de base sur la qualité et la distribution des données.
    
    Returns:
        pd.DataFrame: Jeu de données de transactions générées
    """
    generator = TransactionGenerator(n_customers=1000, n_transactions=10000)
    df = generator.generate_dataset()
    
    # Validation de base
    print("\n=== Tests de Génération des Données ===")
    print(f"Dimensions: {df.shape}")
    print("\nValeurs manquantes:")
    print(df.isnull().sum())
    print("\nDistribution des anomalies:")
    print(df['is_anomaly'].value_counts(normalize=True))
    
    return df

def test_feature_engineering(df):
    """
    Test du pipeline d'ingénierie des caractéristiques.
    
    Traite les données brutes pour générer des caractéristiques
    et valide leur qualité.
    
    Args:
        df (pd.DataFrame): Données de transactions brutes
        
    Returns:
        pd.DataFrame: Données enrichies avec nouvelles caractéristiques
    """
    engineer = TransactionFeatureEngineer()
    df_featured, group_stats = engineer.engineer_features(df)
    
    print("\n=== Tests d'Ingénierie des Caractéristiques ===")
    print(f"Nombre de caractéristiques: {len(df_featured.columns)}")
    
    # Test des valeurs NaN dans les caractéristiques générées
    nan_cols = df_featured.columns[df_featured.isna().any()].tolist()
    if nan_cols:
        print("\nColonnes avec valeurs NaN:")
        print(nan_cols)
    
    return df_featured

def visualize_results(df_featured):
    """
    Création des visualisations de validation.
    
    Génère quatre graphiques pour analyser:
    - Corrélations avec les anomalies
    - Distribution des anomalies par niveau de risque
    - Motifs de transaction
    - Importance des caractéristiques
    
    Args:
        df_featured (pd.DataFrame): Données avec caractéristiques enrichies
    """
    plt.figure(figsize=(15, 10))
    
    # Graphique 1: Corrélations des caractéristiques avec is_anomaly
    plt.subplot(2, 2, 1)
    numeric_cols = df_featured.select_dtypes(include=[np.number]).columns
    correlations = df_featured[numeric_cols].corr()['is_anomaly'].sort_values()
    correlations[-10:].plot(kind='barh')
    plt.title('Top 10 des Caractéristiques Corrélées aux Anomalies')
    
    # Graphique 2: Distribution des anomalies par niveau de risque
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_featured, x='risk_level', y='amount_peer_dev')
    plt.title('Déviation par Rapport aux Pairs selon le Niveau de Risque')
    
    # Graphique 3: Motifs de transaction
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=df_featured.sample(1000),
        x='amount',
        y='amount_velocity_30d',
        hue='is_anomaly',
        alpha=0.6
    )
    plt.title('Motifs de Transaction')
    
    # Graphique 4: Importance des caractéristiques
    plt.subplot(2, 2, 4)
    feature_cols = [col for col in numeric_cols if col.endswith('_dev_30d')]
    avg_values = df_featured[feature_cols].mean().sort_values()
    avg_values.plot(kind='barh')
    plt.title('Moyennes des Caractéristiques de Déviation')
    
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/resultats_validation.png')
    plt.close()

def test_model(df_featured):
    """
    Test du modèle de détection d'anomalies.
    
    Évalue les performances du modèle en:
    - Préparant les caractéristiques
    - Entraînant le modèle
    - Analysant les prédictions
    - Visualisant la matrice de confusion
    
    Args:
        df_featured (pd.DataFrame): Données avec caractéristiques enrichies
        
    Returns:
        np.array: Étiquettes prédites
    """
    print("\n=== Test du Modèle ===")
    
    # Préparation des caractéristiques
    feature_cols = [
        # Caractéristiques basées sur les montants
        'amount_zscore_30d', 'amount_peer_dev', 
        'amount_mean_30d', 'amount_std_30d',
        'amount_roundness', 'near_threshold',
        
        # Caractéristiques de fréquence et vélocité
        'freq_hist_dev_30d', 'amount_velocity_30d',
        'tx_count_30d', 'time_since_last_tx',
        
        # Caractéristiques temporelles
        'outside_business_hours', 'is_weekend', 'is_night',
        
        # Caractéristiques de risque
        'overall_risk_score', 'risk_score', 'structuring_risk'
    ]
    
    # Gestion des valeurs manquantes ou infinies
    X = df_featured[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Ajustement de la contamination selon le taux réel d'anomalies
    actual_anomaly_rate = df_featured['is_anomaly'].mean()
    contamination = max(actual_anomaly_rate, 0.01)  # Contamination minimale de 1%
    
    # Initialisation et entraînement du modèle avec paramètres ajustés
    detector = MLDetector(contamination=contamination)
    detector.fit(X)
    
    # Obtention des prédictions avec seuil ajusté
    scores = detector.isolation_forest.score_samples(X)
    pd.DataFrame(scores).to_csv('scores.csv')
    threshold = np.percentile(scores, contamination * 100)
    pred_labels = np.where(scores < threshold, 1, 0)
    
    # Affichage du rapport de classification
    print("\nRapport de Classification:")
    print(classification_report(df_featured['is_anomaly'], pred_labels))
    
    # Création de la visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df_featured['is_anomaly'], pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Étiquette Réelle')
    plt.xlabel('Étiquette Prédite')
    plt.savefig('/Users/paulconerardy/Documents/Trae/Anomaly v2/data/matrice_confusion.png')
    plt.close()
    
    # Analyse des faux positifs et négatifs
    fp_mask = (df_featured['is_anomaly'] == 0) & (pred_labels == 1)
    fn_mask = (df_featured['is_anomaly'] == 1) & (pred_labels == 0)
    
    print("\nAnalyse des Faux Positifs:")
    print(df_featured[fp_mask][feature_cols].describe())
    
    print("\nAnalyse des Faux Négatifs:")
    print(df_featured[fn_mask][feature_cols].describe())
    
    return pred_labels

def main():
    """
    Fonction principale exécutant le pipeline complet de test.
    
    Étapes:
    - Génération des données
    - Ingénierie des caractéristiques
    - Test du modèle
    - Visualisation des résultats
    - Sauvegarde des données traitées
    """
    # Test de la génération de données
    df = test_data_generation()
    
    # Test de l'ingénierie des caractéristiques
    df_featured = test_feature_engineering(df)
    
    # Test du modèle
    predictions = test_model(df_featured)
    
    # Ajout des prédictions au DataFrame enrichi
    df_featured['predicted_anomaly'] = predictions
    
    # Visualisation des résultats
    visualize_results(df_featured)
    
    # Sauvegarde des données traitées
    output_path = '/Users/paulconerardy/Documents/Trae/Anomaly v2/data/transactions_enrichies.csv'
    df_featured.to_csv(output_path, index=False)
    print(f"\nDonnées traitées sauvegardées dans: {output_path}")

if __name__ == "__main__":
    main()