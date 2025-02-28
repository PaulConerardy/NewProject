import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AMLFeatureEngineer:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        
    def create_features(self, df):
        """Méthode principale pour créer toutes les caractéristiques"""
        df = df.copy()
        df['date_ref'] = pd.to_datetime(df['date_ref'])
        
        df = self._create_amount_features(df)
        df = self._create_time_features(df)
        df = self._create_velocity_features(df)
        df = self._detect_complex_patterns(df)  # Ajouter la détection de motifs complexes
        df = self._create_peer_groups(df)
        df = self._calculate_peer_risk_score(df)
        
        return df
    
    def _detect_complex_patterns(self, df):
        """Détecter les motifs de transaction sophistiqués"""
        # Trier les transactions par date et émetteur
        df = df.sort_values(['emetteur', 'date_ref'])
        
        # Détecter les transactions en succession rapide
        df['time_to_next_tx'] = df.groupby('emetteur')['date_ref'].diff(-1).dt.total_seconds().abs() / 3600
        df['rapid_succession'] = (df['time_to_next_tx'] < 24).astype(int)
        
        # Détecter les transactions aller-retour (même montant)
        df['round_trip'] = ((df['amount'].shift(-1) == df['amount']) & 
                           (df['emetteur'] == df['destinataire'].shift(-1)) & 
                           (df['destinataire'] == df['emetteur'].shift(-1))).astype(int)
        
        # Détecter les motifs de structuration
        df['cumsum_amount'] = df.groupby('emetteur')['amount'].rolling(self.lookback_days).sum().reset_index(0, drop=True)
        df['structured_pattern'] = ((df['cumsum_amount'] > 9000) & 
                                  (df['amount'] < 5000) & 
                                  (df['rapid_succession'] == 1)).astype(int)
        
        # Détecter les motifs de distribution/concentration
        daily_recipients = df.groupby(['emetteur', df['date_ref'].dt.date])['destinataire'].nunique().reset_index()
        daily_recipients.columns = ['emetteur', 'date', 'unique_recipients']
        
        df = df.merge(
            daily_recipients,
            left_on=['emetteur', df['date_ref'].dt.date],
            right_on=['emetteur', 'date'],
            how='left'
        )
        df['fan_out_pattern'] = (df['unique_recipients'] > 3).astype(int)
        
        # Détecter les motifs cycliques
        df['cyclic_pattern'] = ((df['emetteur'] == df['destinataire'].shift(2)) & 
                               (df['amount'].shift(2) * 0.9 <= df['amount']) & 
                               (df['amount'] <= df['amount'].shift(2) * 1.1)).astype(int)
        
        # Calculer le score de risque des motifs
        df['pattern_risk_score'] = (
            df['rapid_succession'] * 2 +
            df['round_trip'] * 3 +
            df['structured_pattern'] * 4 +
            df['fan_out_pattern'] * 2 +
            df['cyclic_pattern'] * 3
        )
        
        return df
    
    def _create_amount_features(self, df):
        """Créer des caractéristiques basées sur les montants"""
        # Les nombres ronds peuvent indiquer des transactions structurées
        df['amount_roundness'] = df['amount'].apply(
            lambda x: len(str(int(x))) - len(str(int(x)).rstrip('0'))
        )
        
        # Transactions juste en dessous des seuils de déclaration
        df['near_threshold'] = (
            ((df['amount'] > 9000) & (df['amount'] < 10000)) |
            ((df['amount'] > 4500) & (df['amount'] < 5000))
        ).astype(int)
        
        # Calculer les statistiques de montant par émetteur
        emetteur_stats = df.groupby('emetteur')['amount'].agg(['mean', 'std']).reset_index()
        df = df.merge(emetteur_stats, on='emetteur', suffixes=('', '_avg'))
        
        # Score Z du montant par rapport à l'historique de l'émetteur
        df['amount_zscore'] = (df['amount'] - df['mean']) / df['std'].fillna(1)
        
        return df
    
    def _create_time_features(self, df):
        """Créer des caractéristiques temporelles"""
        df['hour'] = df['date_ref'].dt.hour
        df['day_of_week'] = df['date_ref'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Grouper les transactions par émetteur et jour
        daily_counts = df.groupby(['emetteur', df['date_ref'].dt.date]).size().reset_index()
        daily_counts.columns = ['emetteur', 'date', 'daily_tx_count']
        
        # Fusionner avec le dataframe original
        df = df.merge(
            daily_counts,
            left_on=['emetteur', df['date_ref'].dt.date],
            right_on=['emetteur', 'date']
        )
        
        return df
    
    def _create_velocity_features(self, df):
        """Créer des caractéristiques liées à la vélocité"""
        # Trier par date
        df = df.sort_values('date_ref')
        
        # Calculer le temps depuis la dernière transaction pour chaque émetteur
        df['time_since_last_tx'] = df.groupby('emetteur')['date_ref'].diff().dt.total_seconds() / 3600
        
        # Calculer le nombre de transactions dans les X derniers jours
        lookback_date = df['date_ref'] - pd.Timedelta(days=self.lookback_days)
        
        tx_counts = df.groupby('emetteur').apply(
            lambda x: x['date_ref'].rolling(self.lookback_days).count()
        ).reset_index(level=0, drop=True)
        
        df['tx_count_30d'] = tx_counts
        
        return df
    
    def _create_peer_groups(self, df):
        """Créer des groupes de pairs basés sur la profession et les motifs de transaction"""
        # Créer des groupes de pairs basés sur la profession
        df['peer_group'] = df['profession']
        
        # Calculer les métriques clés par profession
        peer_metrics = df.groupby('profession').agg({
            'amount': ['mean', 'std', 'median'],
            'risk_level': 'mean',
            'transaction_type': lambda x: x.value_counts().index[0]  # Type de transaction le plus fréquent
        }).reset_index()
        
        # Aplatir les noms de colonnes
        peer_metrics.columns = ['profession', 'peer_avg_amount', 'peer_std_amount', 
                              'peer_median_amount', 'peer_risk_level', 'peer_common_tx_type']
        
        # Fusionner les métriques des pairs avec le dataframe principal
        df = df.merge(peer_metrics, on='profession', how='left')
        
        # Calculer l'écart par rapport au groupe de pairs
        df['amount_deviation_from_peer'] = (df['amount'] - df['peer_avg_amount']) / df['peer_std_amount']
        df['risk_deviation_from_peer'] = df['risk_level'] - df['peer_risk_level']
        
        # Identifier les motifs inhabituels dans le groupe de pairs
        df['unusual_amount_pattern'] = (abs(df['amount_deviation_from_peer']) > 2).astype(int)
        df['unusual_risk_pattern'] = (df['risk_deviation_from_peer'] > 1).astype(int)
        
        # Créer des métriques temporelles pour les pairs (30 derniers jours)
        df['date_ref'] = pd.to_datetime(df['date_ref'])
        lookback_window = df['date_ref'].max() - pd.Timedelta(days=30)
        
        recent_peer_metrics = df[df['date_ref'] >= lookback_window].groupby('profession').agg({
            'amount': ['mean', 'count'],
            'risk_level': 'mean'
        }).reset_index()
        
        recent_peer_metrics.columns = ['profession', 'peer_recent_avg_amount', 
                                     'peer_recent_tx_count', 'peer_recent_risk_level']
        
        df = df.merge(recent_peer_metrics, on='profession', how='left')
        
        # Calculer les écarts récents
        df['recent_amount_deviation'] = (df['amount'] - df['peer_recent_avg_amount']) / df['peer_avg_amount']
        df['recent_risk_deviation'] = df['risk_level'] - df['peer_recent_risk_level']
        
        return df
    
    def _calculate_peer_risk_score(self, df):
        """Calculer le score de risque basé sur les écarts du groupe de pairs"""
        # Initialiser le score de risque
        df['peer_risk_score'] = 0
        
        # Ajouter des points de risque basés sur divers facteurs
        risk_factors = {
            'unusual_amount_pattern': 2,
            'unusual_risk_pattern': 3,
            'amount_deviation_from_peer': lambda x: np.clip(abs(x) - 2, 0, 3),
            'recent_amount_deviation': lambda x: np.clip(abs(x) - 1.5, 0, 2),
            'recent_risk_deviation': lambda x: np.clip(x, 0, 2)
        }
        
        for factor, weight in risk_factors.items():
            if callable(weight):
                df['peer_risk_score'] += weight(df[factor])
            else:
                df['peer_risk_score'] += df[factor] * weight
        
        # Normaliser le score de risque sur une échelle de 0-100
        df['peer_risk_score'] = (df['peer_risk_score'] - df['peer_risk_score'].min()) / \
                               (df['peer_risk_score'].max() - df['peer_risk_score'].min()) * 100
        
        return df