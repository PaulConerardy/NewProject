import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AMLAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Préparer les caractéristiques pour le modèle"""
        feature_columns = [
            # Caractéristiques basées sur les montants
            'amount', 'amount_roundness', 'near_threshold', 'amount_zscore',
            
            # Caractéristiques temporelles
            'is_weekend', 'daily_tx_count',
            
            # Caractéristiques de vélocité
            'time_since_last_tx', 'tx_count_30d',
            
            # Caractéristiques de motifs
            'rapid_succession', 'round_trip', 'structured_pattern',
            'fan_out_pattern', 'cyclic_pattern', 'pattern_risk_score',
            
            # Caractéristiques de groupe de pairs
            'amount_deviation_from_peer', 'risk_deviation_from_peer',
            'unusual_amount_pattern', 'unusual_risk_pattern',
            'recent_amount_deviation', 'recent_risk_deviation',
            'peer_risk_score'
        ]
        
        return df[feature_columns]
    
    def fit(self, df):
        """Entraîner le modèle"""
        X = self.prepare_features(df)
        X = self.scaler.fit_transform(X)
        self.model.fit(X)
        
    def predict(self, df):
        """Prédire les anomalies"""
        X = self.prepare_features(df)
        X = self.scaler.transform(X)
        
        # Obtenir les prédictions brutes (-1 pour anomalies, 1 pour normal)
        predictions = self.model.predict(X)
        
        # Convertir en binaire (1 pour anomalies, 0 pour normal)
        predictions = (predictions == -1).astype(int)
        
        # Obtenir les scores d'anomalie
        scores = self.model.score_samples(X)
        
        return predictions, scores
    
    def plot_predictions(self, df, predictions, scores):
        """Visualiser les prédictions et la performance du modèle"""
        # Créer une figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Graphique 1: Distribution des scores d'anomalie
        sns.histplot(data=scores, bins=50, ax=ax1)
        ax1.set_title('Distribution des Scores d\'Anomalie')
        ax1.set_xlabel('Score d\'Anomalie')
        ax1.set_ylabel('Nombre')
        
        # Graphique 2: Matrice de confusion
        cm = confusion_matrix(df['anomaly'], predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Matrice de Confusion')
        ax2.set_xlabel('Prédit')
        ax2.set_ylabel('Réel')
        ax2.set_xticklabels(['Normal', 'Anomalie'])
        ax2.set_yticklabels(['Normal', 'Anomalie'])
        
        plt.tight_layout()
        plt.savefig('/Users/paulconerardy/Documents/Trae/Anom New Data/data/model_performance.png')
        plt.close()

    def evaluate(self, df):
        """Évaluer la performance du modèle"""
        predictions, scores = self.predict(df)
        
        # Afficher le rapport de classification
        print("\nRapport de Classification:")
        print(classification_report(df['anomaly'], predictions))
        
        # Afficher la matrice de confusion
        print("\nMatrice de Confusion:")
        print(confusion_matrix(df['anomaly'], predictions))
        
        # Générer les visualisations
        self.plot_predictions(df, predictions, scores)
        
        return predictions, scores

# Exemple d'utilisation
if __name__ == "__main__":
    from feature_engineering import AMLFeatureEngineer
    
    # Charger et préparer les données
    df = pd.read_csv("/Users/paulconerardy/Documents/Trae/Anom New Data/data/synthetic_data.csv")
    
    # Créer les caractéristiques
    feature_engineer = AMLFeatureEngineer()
    df_features = feature_engineer.create_features(df)
    
    # Diviser les données en train et test
    train_df = df_features[df_features['date_ref'] < '2023-01-01']
    test_df = df_features[df_features['date_ref'] >= '2023-01-01']
    
    # Initialiser et entraîner le modèle
    detector = AMLAnomalyDetector(contamination=0.1)
    detector.fit(train_df)
    
    # Évaluer sur l'ensemble de test
    predictions, scores = detector.evaluate(test_df)
    
    # Ajouter les prédictions et scores aux données de test
    test_df['predicted_anomaly'] = predictions
    test_df['anomaly_score'] = scores
    
    print("\nVisualisation sauvegardée comme 'model_performance.png'")
    
    # Sauvegarder les résultats
    test_df.to_csv("/Users/paulconerardy/Documents/Trae/Anom New Data/data/predictions.csv", index=False)