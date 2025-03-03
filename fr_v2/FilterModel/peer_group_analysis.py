import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class PeerGroupAnalyzer:
    """
    Analyse les modèles de transaction au sein des groupes de pairs pour détecter
    les anomalies qui pourraient indiquer des activités de blanchiment d'argent.
    """
    
    def __init__(self, data_path):
        """
        Initialise l'analyseur avec les données client.
        
        Paramètres:
        -----------
        data_path : str
            Chemin vers le fichier CSV des données client
        """
        self.data = pd.read_csv(data_path)
        self.preprocessed_data = None
        self.anomaly_scores = None
        self.peer_groups = None
        
    def preprocess_data(self):
        """
        Prétraite les données pour l'analyse:
        - Gère les valeurs manquantes
        - Crée des groupes de pairs
        - Extrait les caractéristiques des transactions
        """
        # Créer une copie des données
        data = self.data.copy()
        
        # Créer des groupes de pairs
        data['peer_group'] = 'Unknown'
        
        # Pour les particuliers, utiliser le groupe de marché
        ind_mask = data['party_type'] == 'Individual'
        data.loc[ind_mask, 'peer_group'] = data.loc[ind_mask, 'mkt_groupe']
        
        # Pour les entreprises, utiliser le code NAICS
        bus_mask = data['party_type'] == 'Business'
        data.loc[bus_mask, 'peer_group'] = data.loc[bus_mask, 'naics_code']
        
        # Extraire les caractéristiques des transactions (toutes les colonnes ACT_PROF)
        txn_cols = [col for col in data.columns if col.startswith('ACT_PROF_')]
        
        # Créer des métriques dérivées qui pourraient être utiles pour la détection
        # 1. Taille moyenne des transactions pour chaque type de transaction
        for txn_type in ['009DJ', '001', '059DJ', 'RECEIVE_BP_INN', '003', '007']:
            vol_col = f'ACT_PROF_{txn_type}_VOL'
            val_col = f'ACT_PROF_{txn_type}_VAL'
            
            # Éviter la division par zéro
            data[f'AVG_{txn_type}_SIZE'] = np.where(
                data[vol_col] > 0,
                data[val_col] / data[vol_col],
                0
            )
        
        # 2. Ratio des transactions entrantes/sortantes
        # En supposant que 001 est entrant et 003 est sortant (ajuster selon vos données)
        data['IN_OUT_RATIO'] = np.where(
            data['ACT_PROF_003_VAL'] > 0,
            data['ACT_PROF_001_VAL'] / data['ACT_PROF_003_VAL'],
            0
        )
        
        # 3. Intensité des transactions par rapport au revenu
        data['TXN_INCOME_RATIO'] = np.where(
            data['income'] > 0,
            (data['ACT_PROF_001_VAL'] + data['ACT_PROF_003_VAL']) / data['income'],
            0
        )
        
        # Stocker les données prétraitées
        self.preprocessed_data = data
        self.peer_groups = data['peer_group'].unique()
        
        return data
    
    def detect_anomalies_by_peer_group(self, method='isolation_forest', contamination=0.05):
        """
        Détecte les anomalies au sein de chaque groupe de pairs en utilisant la méthode spécifiée.
        
        Paramètres:
        -----------
        method : str
            Méthode de détection d'anomalies ('isolation_forest' ou 'lof')
        contamination : float
            Proportion attendue d'anomalies dans le jeu de données
            
        Retourne:
        --------
        DataFrame
            Données originales avec scores d'anomalie et indicateurs
        """
        if self.preprocessed_data is None:
            self.preprocess_data()
        
        data = self.preprocessed_data.copy()
        
        # Caractéristiques à utiliser pour la détection d'anomalies
        feature_cols = [col for col in data.columns if col.startswith('ACT_PROF_') or 
                        col.startswith('AVG_') or 
                        col in ['IN_OUT_RATIO', 'TXN_INCOME_RATIO']]
        
        # Initialiser les scores d'anomalie
        data['anomaly_score'] = 0
        data['is_anomaly'] = False
        
        # Traiter chaque groupe de pairs séparément
        for group in self.peer_groups:
            group_data = data[data['peer_group'] == group].copy()
            
            # Ignorer si trop peu d'échantillons
            if len(group_data) < 10:
                continue
                
            # Extraire les caractéristiques
            X = group_data[feature_cols].fillna(0)
            
            # Standardiser les caractéristiques
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Appliquer la détection d'anomalies
            if method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
                scores = detector.fit_predict(X_scaled)
                # Convertir en scores d'anomalie (plus élevé est plus anormal)
                anomaly_scores = -detector.score_samples(X_scaled)
                
            elif method == 'lof':
                detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                scores = detector.fit_predict(X_scaled)
                # Les scores LOF sont déjà des facteurs d'anomalie
                anomaly_scores = detector.negative_outlier_factor_
            
            # Mettre à jour le DataFrame
            group_indices = group_data.index
            data.loc[group_indices, 'anomaly_score'] = anomaly_scores
            data.loc[group_indices, 'is_anomaly'] = (scores == -1)
        
        # Stocker les résultats
        self.anomaly_scores = data
        
        return data
    
    def visualize_anomalies(self, n_components=2):
        """
        Visualise les anomalies en utilisant l'ACP pour la réduction de dimensionnalité.
        
        Paramètres:
        -----------
        n_components : int
            Nombre de composantes ACP à utiliser pour la visualisation
        """
        if self.anomaly_scores is None:
            self.detect_anomalies_by_peer_group()
        
        data = self.anomaly_scores.copy()
        
        # Caractéristiques à utiliser pour la visualisation
        feature_cols = [col for col in data.columns if col.startswith('ACT_PROF_') or 
                        col.startswith('AVG_') or 
                        col in ['IN_OUT_RATIO', 'TXN_INCOME_RATIO']]
        
        # Créer des sous-graphiques pour chaque groupe de pairs
        unique_groups = data['peer_group'].unique()
        n_groups = len(unique_groups)
        
        # Déterminer la taille de la grille
        grid_size = int(np.ceil(np.sqrt(n_groups)))
        
        plt.figure(figsize=(grid_size*5, grid_size*4))
        
        for i, group in enumerate(unique_groups):
            group_data = data[data['peer_group'] == group].copy()
            
            # Ignorer si trop peu d'échantillons
            if len(group_data) < 10:
                continue
                
            # Extraire les caractéristiques
            X = group_data[feature_cols].fillna(0)
            
            # Standardiser les caractéristiques
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Appliquer l'ACP pour la visualisation
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Créer le sous-graphique
            plt.subplot(grid_size, grid_size, i+1)
            
            # Tracer les points normaux et anormaux
            normal = group_data['is_anomaly'] == False
            anomalous = group_data['is_anomaly'] == True
            
            plt.scatter(X_pca[normal, 0], X_pca[normal, 1], c='blue', alpha=0.5, label='Normal')
            plt.scatter(X_pca[anomalous, 0], X_pca[anomalous, 1], c='red', alpha=0.7, label='Anormal')
            
            plt.title(f'Groupe de pairs: {group}')
            plt.xlabel(f'CP1 ({pca.explained_variance_ratio_[0]:.2f})')
            plt.ylabel(f'CP2 ({pca.explained_variance_ratio_[1]:.2f})')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_top_anomalies(self, n=20):
        """
        Obtient les N principales anomalies à travers tous les groupes de pairs.
        
        Paramètres:
        -----------
        n : int
            Nombre d'anomalies principales à retourner
            
        Retourne:
        --------
        DataFrame
            Les N principales anomalies avec leurs détails
        """
        if self.anomaly_scores is None:
            self.detect_anomalies_by_peer_group()
        
        # Trier par score d'anomalie (décroissant)
        top_anomalies = self.anomaly_scores.sort_values('anomaly_score', ascending=False).head(n)
        
        # Sélectionner les colonnes pertinentes pour l'affichage
        display_cols = ['party_key', 'party_type', 'peer_group', 'income', 
                        'risk_level', 'anomaly_score'] + \
                       [col for col in self.anomaly_scores.columns if col.startswith('ACT_PROF_')]
        
        return top_anomalies[display_cols]
    
    def analyze_anomaly_patterns(self):
        """
        Analyse les modèles dans les anomalies détectées pour identifier les schémas potentiels de blanchiment d'argent.
        
        Retourne:
        --------
        dict
            Dictionnaire contenant les résultats d'analyse
        """
        if self.anomaly_scores is None:
            self.detect_anomalies_by_peer_group()
        
        data = self.anomaly_scores.copy()
        anomalies = data[data['is_anomaly']]
        
        results = {}
        
        # 1. Distribution des anomalies par groupe de pairs
        group_counts = anomalies['peer_group'].value_counts()
        results['anomalies_by_group'] = group_counts
        
        # 2. Distribution des anomalies par niveau de risque
        risk_counts = anomalies['risk_level'].value_counts()
        results['anomalies_by_risk'] = risk_counts
        
        # 3. Métriques moyennes de transaction pour les anomalies vs normal
        txn_cols = [col for col in data.columns if col.startswith('ACT_PROF_')]
        avg_metrics = pd.DataFrame({
            'anomalous': anomalies[txn_cols].mean(),
            'normal': data[~data['is_anomaly']][txn_cols].mean(),
            'ratio': anomalies[txn_cols].mean() / data[~data['is_anomaly']][txn_cols].mean()
        })
        results['avg_metrics'] = avg_metrics
        
        # 4. Identifier les schémas potentiels de banque clandestine
        # Rechercher des flux équilibrés entre entités (montants entrants et sortants similaires)
        anomalies['flow_balance'] = np.abs(
            anomalies['ACT_PROF_001_VAL'] - anomalies['ACT_PROF_003_VAL']
        ) / (anomalies['ACT_PROF_001_VAL'] + anomalies['ACT_PROF_003_VAL'] + 1)
        
        potential_underground = anomalies[anomalies['flow_balance'] < 0.2].sort_values('flow_balance')
        results['potential_underground_banking'] = potential_underground
        
        # 5. Identifier le potentiel blanchiment d'argent basé sur le commerce
        # Volume élevé de transactions internationales par rapport au type d'entreprise
        anomalies['intl_txn_ratio'] = anomalies['ACT_PROF_RECEIVE_BP_INN_VAL'] / (anomalies['income'] + 1)
        potential_trade_ml = anomalies.sort_values('intl_txn_ratio', ascending=False).head(20)
        results['potential_trade_ml'] = potential_trade_ml
        
        return results