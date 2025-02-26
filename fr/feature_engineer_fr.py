import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from features.network_features import TransactionNetworkAnalyzer

class TransactionFeatureEngineer:
    """
    Classe pour l'ingénierie des caractéristiques des transactions financières.
    
    Cette classe génère des caractéristiques avancées pour la détection d'anomalies
    dans les transactions financières, incluant:
    - Caractéristiques temporelles
    - Analyses des montants
    - Métriques de vélocité
    - Détection de motifs
    - Analyses de réseau
    - Évaluation des risques
    
    Attributs:
        lookback_days (int): Période d'historique en jours pour l'analyse
    """
    
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
    
    def engineer_features(self, df):
        """
        Fonction principale de génération des caractéristiques.
        
        Transforme les données brutes en caractéristiques exploitables pour
        la détection d'anomalies.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les transactions brutes
            
        Returns:
            tuple: (DataFrame enrichi, statistiques des groupes)
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Caractéristiques temporelles
        df = self._create_time_features(df)
        
        # Caractéristiques basées sur les montants
        df = self._create_amount_features(df)
        
        # Caractéristiques de vélocité et fréquence
        df = self._create_velocity_features(df)
        
        # Caractéristiques de motifs
        df = self._create_pattern_features(df)
        
        # Ajout des caractéristiques de réseau
        network_analyzer = TransactionNetworkAnalyzer(lookback_days=self.lookback_days)
        df = network_analyzer.create_network_features(df)
        df.to_csv('df_network_features.csv')
        
        # Caractéristiques de risque
        df = self._create_risk_features(df)
        
        # Caractéristiques de déviation historique
        df = self._create_historical_deviation_features(df)

        # Création et validation des groupes de pairs
        df = self._create_peer_groups(df)
        group_stats = self._validate_peer_groups(df)
        
        # Caractéristiques de déviation par rapport aux pairs
        df, group_stats = self._create_peer_deviation_features(df)
        
        return df, group_stats

    def _create_time_features(self, df):
        """
        Génère les caractéristiques temporelles.
        
        Crée des indicateurs pour:
        - Heure de la journée
        - Jour de la semaine
        - Week-end
        - Transactions nocturnes
        - Heures ouvrables
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques temporelles
        """
        # Composants temporels de base
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        
        # Fenêtres temporelles pour agrégations
        df['date'] = df['timestamp'].dt.date
        
        # Déviation des heures ouvrables
        business_hours = (df['hour'] >= 9) & (df['hour'] <= 17) & (~df['is_weekend'])
        df['outside_business_hours'] = (~business_hours).astype(int)
        
        return df

    def _create_amount_features(self, df):
        """
        Génère les caractéristiques basées sur les montants.
        
        Calcule des statistiques glissantes sur les montants:
        - Moyennes sur différentes périodes
        - Écarts-types
        - Scores Z
        - Détection des montants arrondis
        - Détection des seuils réglementaires
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques de montants
        """
        df = df.copy()
        df = df.set_index('timestamp').sort_index()
        
        for window in ['1D', '7D', '30D']:
            window_str = window.replace('D', 'd')
            
            # Calcul des statistiques pour chaque référence
            for ref_id in df.index.get_level_values(0).unique():
                mask = df.index.get_level_values(0) == ref_id
                
                # Calcul des statistiques glissantes
                df.loc[mask, f'amount_mean_{window_str}'] = df.loc[mask, 'amount'].rolling(
                    window=window,
                    min_periods=1
                ).mean()
                
                df.loc[mask, f'amount_std_{window_str}'] = df.loc[mask, 'amount'].rolling(
                    window=window,
                    min_periods=1
                ).std().fillna(1)
                
                # Calcul du score Z
                df.loc[mask, f'amount_zscore_{window_str}'] = (
                    df.loc[mask, 'amount'] - df.loc[mask, f'amount_mean_{window_str}']
                ) / df.loc[mask, f'amount_std_{window_str}']
        
        df = df.reset_index()
        
        # Détection des montants arrondis
        df['amount_roundness'] = df['amount'].apply(
            lambda x: len(str(int(x))) - len(str(int(x)).rstrip('0'))
        )
        
        # Détection des montants proches des seuils
        df['near_threshold'] = (
            ((df['amount'] > 9000) & (df['amount'] < 10000)) |
            ((df['amount'] > 4500) & (df['amount'] < 5000))
        ).astype(int)
        
        return df

    def _create_velocity_features(self, df):
        """
        Génère les caractéristiques de vélocité et fréquence.
        
        Calcule:
        - Nombre de transactions par période
        - Vélocité des montants
        - Temps entre les transactions
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques de vélocité
        """
        df = df.set_index('timestamp').sort_index()
        
        # Fréquence des transactions
        for window in [1, 7, 30]:
            # Comptage des transactions
            freq = df.groupby('ref_id').rolling(
                window=f'{window}D'
            )['amount'].count().reset_index()
            
            df[f'tx_count_{window}d'] = freq['amount']
            
            # Calcul de la vélocité des montants
            amount_sum = df.groupby('ref_id')['amount'].rolling(
                window=f'{window}D'
            ).sum().reset_index()
            
            df[f'amount_velocity_{window}d'] = amount_sum['amount'] / window
        
        df = df.reset_index()
        
        # Temps entre les transactions
        df['time_since_last_tx'] = df.groupby('ref_id')['timestamp'].diff().dt.total_seconds() / 3600
        
        return df

    def _create_pattern_features(self, df):
        """
        Génère les caractéristiques basées sur les motifs comportementaux.
        
        Détecte:
        - Motifs de montants répétitifs
        - Schémas de transactions similaires
        - Comportements de structuration
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques de motifs
        """
        df = df.set_index('timestamp').sort_index()
        
        # Motifs répétitifs
        for window in [7, 30]:
            # Motifs de montants identiques
            same_amount = df.groupby('ref_id').rolling(
                window=f'{window}D'
            )['amount'].apply(
                lambda x: (x.nunique() == 1) and (len(x) > 1)
            ).reset_index()
            df[f'same_amount_pattern_{window}d'] = same_amount['amount'].astype(int)
        
        df = df.reset_index()
        
        # Motifs de structuration
        df['structuring_risk'] = (
            (df['near_threshold'] == 1) &
            (df['time_since_last_tx'] < 48)
        ).astype(int)
        
        return df
    
    def _create_network_features(self, df):
        """
        Génère les caractéristiques basées sur l'analyse de réseau.
        
        Analyse:
        - Réseaux de pays destinataires
        - Ratios de pays à risque
        - Interconnexions des transactions
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques de réseau
        """
        df = df.set_index('timestamp').sort_index()
        
        # Caractéristiques du réseau de pays
        country_stats = df.groupby('ref_id').rolling(
            window='30D'
        ).agg({
            'recipient_country': lambda x: x.nunique()
        }).reset_index()
        df['unique_countries_30d'] = country_stats['recipient_country']
        
        # Ratio des pays à haut risque
        pays_risques = ['CN']  # Exemple de pays à haut risque
        risk_stats = df.groupby('ref_id').rolling(
            window='30D'
        ).agg({
            'recipient_country': lambda x: (x.isin(pays_risques)).mean()
        }).reset_index()
        df['high_risk_country_ratio_30d'] = risk_stats['recipient_country']
        
        df = df.reset_index()
        
        return df
    
    def _create_risk_features(self, df):
        """
        Génère les caractéristiques liées au risque.
        
        Combine plusieurs facteurs de risque:
        - Score de risque de base
        - Risque de structuration
        - Risque temporel
        - Risque lié aux montants
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec scores de risque agrégés
        """
        # Combinaison des facteurs de risque
        df['overall_risk_score'] = (
            df['risk_score'] * 0.3 +
            df['structuring_risk'] * 0.2 +
            df['outside_business_hours'] * 0.1 +
            df['amount_roundness'] * 0.1 +
            df['near_threshold'] * 0.1
        )
        
        # Catégories de niveau de risque
        df['risk_level'] = pd.qcut(
            df['overall_risk_score'],
            q=5,
            labels=['très_faible', 'faible', 'moyen', 'élevé', 'très_élevé']
        )
        
        return df
    
    def _create_historical_deviation_features(self, df):
        """
        Génère les caractéristiques de déviation historique.
        
        Analyse les écarts par rapport au comportement historique:
        - Déviations des montants
        - Déviations de fréquence
        - Changements de comportement
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec caractéristiques de déviation
        """
        df_original = df.copy()
        df = df.set_index(['timestamp', 'ref_id']).sort_index()
        
        # Calcul des périodes de référence
        for window in [7, 30]:
            # Déviation par rapport à la moyenne historique
            hist_amount_stats = df.groupby(level=1)['amount'].rolling(
                window=window, min_periods=1
            ).agg(['mean', 'std'])
            hist_amount_stats.to_csv(f'hist_amount_stats_{window}.csv')
            
            hist_amount_stats = hist_amount_stats.reset_index(level=0)
            df = df.join(hist_amount_stats, rsuffix='_hist')
            
            # Calcul des déviations
            df[f'amount_hist_dev_{window}d'] = (
                df['amount'] - df['mean']
            ) / df['std'].replace(0, 1)
            
            df = df.drop(['mean', 'std'], axis=1)
            
            # Déviation de fréquence
            hist_freq = df.groupby(level=1)['amount'].rolling(
                window=window
            ).count()
            hist_freq = hist_freq.reset_index(level=0)
            hist_freq.name = f'count_{window}d'
            hist_freq = hist_freq.rename(columns={'amount': f'count_{window}d'})
            
            df = df.join(hist_freq, rsuffix='_hist')
            
            # Fréquence actuelle
            current_freq = df.groupby(level=1)['amount'].rolling(1).count()
            current_freq = current_freq.reset_index(level=0)
            current_freq.name = f'current_count_{window}d'
            current_freq = current_freq.rename(columns={'amount': f'current_count_{window}d'})
            
            df = df.join(current_freq, rsuffix='_hist')
            
            # Calcul de la déviation de fréquence
            df[f'freq_hist_dev_{window}d'] = (
                df[f'current_count_{window}d'] - df[f'count_{window}d']
            ) / df[f'count_{window}d'].replace(0, 1)
            
            df = df.drop([f'count_{window}d', f'current_count_{window}d'], axis=1)
# Réinitialisation de l'index pour le traitement ultérieur
        df = df.reset_index(drop=True)
        
        # Copie des nouvelles caractéristiques vers le DataFrame original
        new_features = set([col for col in df.columns if col not in df_original.columns])
        df = df.loc[:,~df.columns.duplicated()].copy()
        
        for feature in new_features:
            df_original[feature] = df[feature].values
        
        return df_original
    
    def _create_peer_groups(self, df):
        """
        Crée des groupes de pairs sophistiqués basés sur plusieurs caractéristiques.
        
        Analyse:
        - Montants moyens mensuels
        - Fréquence des transactions
        - Segments d'activité
        - Comportements similaires
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec groupes de pairs assignés
        """
        # Calcul des métriques comportementales
        df['avg_monthly_amount'] = df.groupby('ref_id')['amount'].transform('mean')
        
        # Calcul de la fréquence avec gestion des valeurs infinies
        time_diff = (df.groupby('ref_id')['timestamp'].transform('max') - 
                    df.groupby('ref_id')['timestamp'].transform('min')).dt.days
        
        # Gestion des différences temporelles nulles
        time_diff = time_diff.replace(0, 1)
        
        df['tx_frequency'] = df.groupby('ref_id').size() / time_diff
        
        # Création des segments d'activité
        labels = ['TF', 'F', 'M', 'E', 'TE']  # Très Faible à Très Élevé
        df['amount_segment'] = pd.qcut(
            df['avg_monthly_amount'],
            q=5,
            labels=labels
        )
        df['amount_segment'] = df['amount_segment'].cat.add_categories(['Inconnu']).fillna('Inconnu')
        
        # Segments de fréquence
        freq_labels = ['Faible', 'Moyen', 'Élevé']
        df['frequency_segment'] = pd.qcut(
            df['tx_frequency'],
            q=3,
            labels=freq_labels
        )
        df['frequency_segment'] = df['frequency_segment'].cat.add_categories(['Inconnu']).fillna('Inconnu')
        
        # Définition des groupes de pairs
        df['peer_group'] = df.apply(self._assign_peer_group, axis=1)
        
        return df

    def _assign_peer_group(self, row):
        """
        Assigne un groupe de pairs à une transaction.
        
        Utilise une combinaison de:
        - Segment de montant
        - Segment de fréquence
        - Type de client
        - Secteur d'activité
        
        Args:
            row (pd.Series): Ligne de transaction
            
        Returns:
            str: Identifiant du groupe de pairs
        """
        return f"{row['amount_segment']}_{row['frequency_segment']}"
    
    def _validate_peer_groups(self, df):
        """
        Valide la cohérence des groupes de pairs.
        
        Vérifie:
        - Taille minimale des groupes
        - Homogénéité des comportements
        - Stabilité temporelle
        
        Args:
            df (pd.DataFrame): DataFrame avec groupes assignés
            
        Returns:
            dict: Statistiques de validation des groupes
        """
        group_stats = {}
        
        # Analyse de la taille des groupes
        size_stats = df.groupby('peer_group').size()
        group_stats['group_sizes'] = size_stats.to_dict()
        
        # Analyse de l'homogénéité
        amount_stats = df.groupby('peer_group')['amount'].agg(['mean', 'std']).to_dict()
        group_stats['amount_stats'] = amount_stats
        
        # Analyse de la stabilité temporelle
        time_stats = df.groupby(['peer_group', pd.Grouper(key='timestamp', freq='M')]).size()
        group_stats['temporal_stability'] = time_stats.to_dict()
        
        return group_stats
    
    def _create_peer_deviation_features(self, df):
        """
        Calcule les déviations par rapport aux groupes de pairs.
        
        Analyse:
        - Écarts de montants
        - Écarts de fréquence
        - Comportements atypiques
        
        Args:
            df (pd.DataFrame): DataFrame avec groupes de pairs
            
        Returns:
            tuple: (DataFrame enrichi, statistiques des groupes)
        """
        # Statistiques par groupe de pairs
        peer_stats = df.groupby('peer_group').agg({
            'amount': ['mean', 'std'],
            'tx_frequency': ['mean', 'std']
        })
        
        # Calcul des déviations
        for group in df['peer_group'].unique():
            mask = df['peer_group'] == group
            
            # Déviation des montants
            df.loc[mask, 'amount_peer_dev'] = (
                df.loc[mask, 'amount'] - peer_stats.loc[group, ('amount', 'mean')]
            ) / peer_stats.loc[group, ('amount', 'std')].clip(lower=1)
            
            # Déviation de fréquence
            df.loc[mask, 'frequency_peer_dev'] = (
                df.loc[mask, 'tx_frequency'] - peer_stats.loc[group, ('tx_frequency', 'mean')]
            ) / peer_stats.loc[group, ('tx_frequency', 'std')].clip(lower=1)
        
        return df, peer_stats.to_dict()