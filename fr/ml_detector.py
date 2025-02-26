from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURE_COLS, RISK_THRESHOLDS
from sklearn.preprocessing import RobustScaler

class MLDetector:
    """
    Détecteur d'anomalies basé sur l'apprentissage automatique.
    
    Cette classe implémente un système de détection d'anomalies utilisant:
    - Isolation Forest pour la détection non supervisée
    - Prétraitement robuste des données
    - Gestion adaptative des seuils
    - Combinaison de méthodes statistiques et basées sur des règles
    
    Attributs:
        contamination (float): Proportion attendue d'anomalies dans les données
    """
    
    def __init__(self, contamination=0.1):
        # Ajustement des paramètres pour une meilleure stabilité numérique
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=500,
            max_samples='auto',
            random_state=42,
            bootstrap=True,
            n_jobs=-1,
            max_features=1.0  # Utilisation de toutes les caractéristiques
        )
        
        # Utilisation de RobustScaler pour une meilleure gestion des valeurs aberrantes
        self.scaler = RobustScaler(quantile_range=(1, 99))
        self.feature_cols = FEATURE_COLS
        self.fitted_features = None  # Suivi des caractéristiques ajustées

    def fit(self, X):
        """
        Entraîne le modèle de détection d'anomalies avec prétraitement amélioré.
        
        Étapes:
        - Sélection des caractéristiques spécifiées
        - Gestion des valeurs manquantes
        - Normalisation robuste
        - Entraînement de l'Isolation Forest
        
        Args:
            X (pd.DataFrame): Données d'entraînement
            
        Returns:
            self: Instance entraînée du détecteur
        """
        # Utilisation uniquement des caractéristiques spécifiées
        X = X[self.feature_cols].copy()
        
        # Gestion des valeurs manquantes en premier
        X = self._handle_nan_values(X)
        X = self._preprocess_features(X)
        
        # Conversion en tableau numpy
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement de l'Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        return self

    def predict(self, X):
        """
        Prédit les anomalies avec prétraitement amélioré.
        
        Processus:
        - Prétraitement des données
        - Calcul des scores d'anomalie
        - Application du seuil dynamique
        
        Args:
            X (pd.DataFrame): Données à évaluer
            
        Returns:
            np.array: Prédictions (-1 pour anomalie, 1 pour normal)
        """
        X = X[self.feature_cols].copy()
        
        # Gestion des valeurs manquantes
        X = self._handle_nan_values(X)
        X = self._preprocess_features(X)
        
        # Conversion en tableau numpy
        X_scaled = self.scaler.transform(X)
        
        # Calcul des scores
        scores = self.isolation_forest.score_samples(X_scaled)
        scores.to_csv('scores.csv')
        threshold = self._calculate_dynamic_threshold(scores)
        
        return np.where(scores < threshold, -1, 1)

    def _handle_nan_values(self, X):
        """
        Gère les valeurs manquantes selon le type de caractéristique.
        
        Stratégies:
        - Colonnes numériques: médiane mobile puis médiane globale
        - Colonnes catégorielles: mode
        
        Args:
            X (pd.DataFrame): Données avec valeurs manquantes
            
        Returns:
            pd.DataFrame: Données avec valeurs manquantes traitées
        """
        for column in X.columns:
            if X[column].isnull().any():
                if X[column].dtype in ['float64', 'int64']:
                    # Pour les colonnes numériques
                    X[column] = X[column].fillna(
                        X[column].rolling(window=3, min_periods=1).median()
                    )
                    # Si toujours NaN, utiliser la médiane globale
                    X[column] = X[column].fillna(X[column].median())
                else:
                    # Pour les colonnes catégorielles
                    X[column] = X[column].fillna(X[column].mode()[0])
        
        return X
    
    def _apply_rules(self, X):
        """
        Applique la détection basée sur des règles.
        
        Méthode:
        - Vérification des scores Z
        - Agrégation des violations de règles
        
        Args:
            X (pd.DataFrame): Données à analyser
            
        Returns:
            np.array: Scores basés sur les règles
        """
        # Implémentation des vérifications de base
        rule_scores = np.zeros(len(X))
        
        # Ajout direct des scores basés sur les règles
        for col in X.columns:
            if col.endswith('_zscore'):
                rule_scores += np.abs(X[col])
            
        return rule_scores / len(X.columns)

    def _preprocess_features(self, X):
        """
        Prétraite les caractéristiques pour gérer les valeurs manquantes et infinies.
        
        Processus:
        - Remplacement des valeurs infinies
        - Traitement spécifique par type de caractéristique
        - Gestion des distributions asymétriques
        
        Args:
            X (pd.DataFrame): Données à prétraiter
            
        Returns:
            pd.DataFrame: Données prétraitées
        """
        # Remplacement des valeurs infinies
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Gestion des valeurs manquantes par type de caractéristique
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_features:
            if X[col].isnull().any():
                # Utilisation de la médiane pour les caractéristiques très asymétriques
                if abs(X[col].skew()) > 1:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mean())
        
        return X
    
    def _calculate_dynamic_threshold(self, scores):
        """
        Calcule le seuil en utilisant des statistiques robustes.
        
        Méthode:
        - Calcul basé sur l'écart interquartile (IQR)
        - Ajustement pour contamination minimale
        - Gestion adaptative des seuils
        
        Args:
            scores (np.array): Scores d'anomalie
            
        Returns:
            float: Seuil calculé
        """
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        # Utilisation du seuil basé sur l'IQR
        threshold = q1 - 1.5 * iqr
        
        # Assurer une contamination minimale
        min_anomalies = int(len(scores) * self.isolation_forest.contamination)
        if (scores < threshold).sum() < min_anomalies:
            threshold = np.percentile(scores, self.isolation_forest.contamination * 100)
            
        return threshold

    def _combine_scores(self, if_scores, rule_scores):
        """
        Combine différentes méthodes de détection avec des poids ajustés.
        
        Stratégie:
        - Normalisation des scores
        - Pondération adaptative
        - Fusion des méthodes
        
        Args:
            if_scores (np.array): Scores de l'Isolation Forest
            rule_scores (np.array): Scores basés sur les règles
            
        Returns:
            np.array: Scores combinés
        """
        if_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        # Ajustement des poids pour donner plus d'importance à la détection basée sur les règles
        weights = {
            'isolation_forest': 0.4,  # Réduit de 0.6
            'rules': 0.6             # Augmenté de 0.4
        }
        
        return (weights['isolation_forest'] * if_normalized + 
                weights['rules'] * rule_scores)