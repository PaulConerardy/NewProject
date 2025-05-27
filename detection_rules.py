import pandas as pd
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod
import json # Needed for potentially parsing triggered rules details
from datetime import timedelta

class DetectionRule(ABC):
    """Classe de base abstraite pour les règles de détection."""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        # Assurer que les seuils et scores spécifiques à la règle existent dans la configuration
        if name in config.get('rules', {}):
             self.rule_config = config['rules'][name]
        else:
             self.rule_config = {'thresholds': [], 'scores': []}
             print(f"Avertissement : Configuration pour la règle '{name}' introuvable. Utilisation de la configuration par défaut vide.")

    @abstractmethod
    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        """
        Appliquer la règle de détection à une entité.

        Args:
            entity_id (Any): L'ID de l'entité évaluée.
            transactions_df (pd.DataFrame | None): DataFrame contenant les données de transaction.
            wires_df (pd.DataFrame | None): DataFrame contenant les données de virement bancaire.
            entity_df (pd.DataFrame | None): DataFrame contenant les données d'entité.

        Returns:
            Tuple[int, Dict]: Un tuple contenant le score pour cette règle et un dictionnaire
                                 avec les détails de ce qui a déclenché la règle (le cas échéant).
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Retourne une brève description de la règle.
        """
        pass

    def _get_graduated_score(self, count: int) -> int:
        """
        Calculer le score basé sur le compteur et les seuils/scores gradués de la règle.
        """
        thresholds = self.rule_config.get('thresholds', [])
        scores = self.rule_config.get('scores', [])

        for i in range(len(thresholds) - 1, -1, -1):
            if count >= thresholds[i]:
                return scores[i]

        return 0

    def _convert_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Méthode auxiliaire pour convertir les colonnes de dates en toute sécurité.
        """
        if df is None or df.empty or date_column not in df.columns: # Handle None, empty, or missing column
             return df
        
        df = df.copy()
        try:
            # Essayer plusieurs formats de date, y compris celui du code original
            df[date_column] = pd.to_datetime(df[date_column], format='%d%b%Y', errors='coerce')
            if df[date_column].isna().all() and not df[date_column].empty: # Si tous ont échoué, essayer d'inférer le format
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception:
            # Repli si les formats spécifiques échouent
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
        return df


class LargeWireTransferRule(DetectionRule):
    """
    Règle pour détecter les virements importants ou les dépôts en espèces suivis de transferts sortants.
    """

    def __init__(self, config: Dict):
        super().__init__('large_wire_transfers_followed_by_outgoing', config)
        # Paramètres spécifiques à la règle issus de la configuration, avec des valeurs par défaut
        self.large_amount_threshold = self.config['detection_params'].get('large_amount_threshold', 5000)
        self.subsequent_transfer_count_threshold = self.config['detection_params'].get('subsequent_transfer_count_threshold', 3)
        self.subsequent_transfer_time_window_days = self.config['detection_params'].get('subsequent_transfer_time_window_days', 3)

    def get_description(self) -> str:
        return "Reçoit soudainement des virements importants ou dépôts suivis de transferts sortants"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}

        # Vérifier les virements entrants importants
        if wires_df is not None and not wires_df.empty:
            incoming_wires = wires_df[(wires_df['party_key'] == entity_id) & (wires_df['sign'] == '+')].copy()
            large_wires = incoming_wires[incoming_wires['amount'] > self.large_amount_threshold]

            if not large_wires.empty:
                count += 1
                details['large_incoming_wires'] = large_wires[['wire_key', 'amount', 'wire_date']].to_dict(orient='records')

                # Vérifier les transactions sortantes ultérieures
                if transactions_df is not None and not transactions_df.empty:
                    large_wires = self._convert_dates(large_wires, 'wire_date')
                    transactions_df_dated = self._convert_dates(transactions_df, 'trx_date')

                    if not large_wires.empty and not transactions_df_dated.empty:
                         min_date = large_wires['wire_date'].min()
                         time_window_end = min_date + pd.Timedelta(days=self.subsequent_transfer_time_window_days)

                         outgoing_txns = transactions_df_dated[
                             (transactions_df_dated['party_key'] == entity_id) &
                             (transactions_df_dated['sign'] == '-') &
                             (transactions_df_dated['trx_date'] > min_date) &
                             (transactions_df_dated['trx_date'] <= time_window_end) # Ajout de la vérification de la fin de la fenêtre temporelle
                         ].copy()

                         if len(outgoing_txns) > self.subsequent_transfer_count_threshold:
                             count += 1
                             details['subsequent_outgoing_txns'] = outgoing_txns[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')

                             unique_types = outgoing_txns['transaction_type_desc'].nunique()
                             if unique_types > 1: # Supposons que cette vérification de la variété des types de transaction est toujours pertinente
                                 count += 1
                                 details['subsequent_outgoing_txns_variety'] = unique_types

        # Vérifier également les gros dépôts en espèces
        if transactions_df is not None and not transactions_df.empty:
            transactions_df_dated = self._convert_dates(transactions_df, 'trx_date')
            cash_deposits = transactions_df_dated[
                (transactions_df_dated['party_key'] == entity_id) &
                (transactions_df_dated['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df_dated['sign'] == '+')
            ].copy()

            large_deposits = cash_deposits[cash_deposits['amount'] > self.large_amount_threshold]

            if not large_deposits.empty:
                count += 1
                details['large_cash_deposits'] = large_deposits[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')

                # Vérifier les transactions sortantes ultérieures après les gros dépôts
                if not large_deposits.empty:
                     min_date = large_deposits['trx_date'].min()
                     time_window_end = min_date + pd.Timedelta(days=self.subsequent_transfer_time_window_days)

                     outgoing_txns = transactions_df_dated[
                         (transactions_df_dated['party_key'] == entity_id) &
                         (transactions_df_dated['sign'] == '-') &
                         (transactions_df_dated['trx_date'] > min_date) &
                         (transactions_df_dated['trx_date'] <= time_window_end) # Ajout de la vérification de la fin de la fenêtre temporelle
                     ].copy()

                     if len(outgoing_txns) > self.subsequent_transfer_count_threshold:
                         count += 1
                         # Éviter de dupliquer les détails si déjà ajoutés par la vérification des gros virements
                         if 'subsequent_outgoing_txns' not in details:
                              details['subsequent_outgoing_txns'] = outgoing_txns[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')
                         if 'subsequent_outgoing_txns_variety' not in details:
                             unique_types = outgoing_txns['transaction_type_desc'].nunique()
                             if unique_types > 1:
                                 count += 1
                                 details['subsequent_outgoing_txns_variety'] = unique_types

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class SanctionedCountriesRule(DetectionRule):
    """
    Règle pour détecter les transactions avec des pays sanctionnés ou à haut risque.
    """
    def __init__(self, config: Dict):
        super().__init__('sanctioned_countries_transactions', config)
        self.suspicious_countries = self.config['detection_params'].get('suspicious_countries_list', ['CN', 'HK', 'AE', 'IR', 'KW'])

    def get_description(self) -> str:
        return "Transactions avec des pays sanctionnés ou à haut risque"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}
        triggered_wires = []

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if not entity_wires.empty:
                suspicious_origin = entity_wires[entity_wires['originator_country'].isin(self.suspicious_countries)].copy()
                suspicious_dest = entity_wires[entity_wires['beneficiary_country'].isin(self.suspicious_countries)].copy()

                count = len(suspicious_origin) + len(suspicious_dest)

                if not suspicious_origin.empty:
                    triggered_wires.extend(suspicious_origin[['wire_key', 'originator_country', 'beneficiary_country']].to_dict(orient='records'))
                if not suspicious_dest.empty:
                     triggered_wires.extend(suspicious_dest[['wire_key', 'originator_country', 'beneficiary_country']].to_dict(orient='records'))

        details['triggered_wires'] = triggered_wires # Peut être vide

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class SplitCashDepositsRule(DetectionRule):
    """
    Règle pour détecter les dépôts en espèces fractionnés juste en dessous du seuil de déclaration le même jour.
    """
    def __init__(self, config: Dict):
        super().__init__('split_cash_deposits_same_day', config)
        # Supposons que threshold_amount est pertinent ici pour la vérification du montant
        self.threshold_amount = self.config['detection_params'].get('threshold_amount', 10000)
        # Supposons qu'un seuil pour le nombre de succursales impliquées ou le montant total quotidien pourrait être pertinent
        # Basé sur la logique originale, il semble vérifier les dépôts > 5000 et provenant de > 1 succursale
        self.daily_amount_threshold = self.config['detection_params'].get('split_deposit_daily_amount_threshold', 5000)
        self.min_branches = self.config['detection_params'].get('split_deposit_min_branches', 1) # Le code original vérifiait > 1 succursale

    def get_description(self) -> str:
        return "Dépôts en espèces fractionnés le même jour à plusieurs endroits"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}
        suspicious_days_list = []

        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts en espèces pour cette entité
            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df['sign'] == '+')
            ].copy()

            if not cash_deposits.empty:
                # Convertir les dates et supprimer les valeurs NaT
                cash_deposits = self._convert_dates(cash_deposits, 'trx_date')
                cash_deposits = cash_deposits.dropna(subset=['trx_date'])

                if not cash_deposits.empty:
                    # Grouper par date et vérifier plusieurs succursales/montant total élevé
                    daily_deposits = cash_deposits.groupby(cash_deposits['trx_date'].dt.date).agg(
                        branch_count=('branch', 'nunique'),
                        total_amount=('amount', 'sum')
                    ).reset_index()
                    # Renommer la colonne de date pour plus de clarté après le groupement
                    daily_deposits = daily_deposits.rename(columns={'trx_date': 'date'})

                    # Identifier les jours suspects en fonction des critères
                    # Logique ajustée pour vérifier branch_count > self.min_branches et total_amount > self.daily_amount_threshold
                    suspicious_days = daily_deposits[
                        (daily_deposits['branch_count'] > self.min_branches) &
                        (daily_deposits['total_amount'] > self.daily_amount_threshold)
                    ]

                    count = len(suspicious_days)

                    if not suspicious_days.empty:
                         # Inclure les détails sur les jours suspects
                         # Convertir les objets date en chaîne pour la sérialisation JSON
                         suspicious_days_list = suspicious_days.to_dict(orient='records')
                         for day_detail in suspicious_days_list:
                             day_detail['date'] = str(day_detail['date'])
                         details['suspicious_days'] = suspicious_days_list

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class SuspectedMoneyMulesRule(DetectionRule):
    """
    Règle pour détecter les mules financières suspectées en fonction des modèles de transaction.
    """
    def __init__(self, config: Dict):
        super().__init__('suspected_money_mules', config)
        # Extraire les paramètres spécifiques à la règle de la configuration
        self.individual_deposit_threshold = self.config['detection_params'].get('mule_individual_deposit_threshold', 5)
        self.individual_email_transfer_threshold = self.config['detection_params'].get('mule_individual_email_transfer_threshold', 5)
        self.mule_withdrawal_time_window_days = self.config['detection_params'].get('mule_withdrawal_time_window_days', 3)
        self.mule_individual_wire_threshold = self.config['detection_params'].get('mule_individual_wire_threshold', 2)

    def get_description(self) -> str:
        return "Utilisation de mules financières suspectées"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]
                details['entity_type'] = entity_type

        if transactions_df is not None and not transactions_df.empty:
            incoming_txns = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            if not incoming_txns.empty:
                deposit_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Depot Especes'])
                email_transfer_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Transfert Internet'])

                if entity_type == 'Particulier':
                    if deposit_count > self.individual_deposit_threshold:
                        count += 1
                        details['individual_deposit_count'] = deposit_count
                    if email_transfer_count > self.individual_email_transfer_threshold:
                        count += 1
                        details['individual_email_transfer_count'] = email_transfer_count

                outgoing_txns = transactions_df[
                    (transactions_df['party_key'] == entity_id) &
                    (transactions_df['sign'] == '-')
                ].copy()

                if not outgoing_txns.empty and not incoming_txns.empty:
                    incoming_txns = self._convert_dates(incoming_txns, 'trx_date')
                    outgoing_txns = self._convert_dates(outgoing_txns, 'trx_date')

                    # Assurer que les dates sont valides avant de trouver min/max
                    incoming_txns_valid_dates = incoming_txns.dropna(subset=['trx_date'])
                    outgoing_txns_valid_dates = outgoing_txns.dropna(subset=['trx_date'])

                    if not incoming_txns_valid_dates.empty and not outgoing_txns_valid_dates.empty:
                        # Regarder le premier dépôt et le dernier retrait
                        first_deposit_date = incoming_txns_valid_dates['trx_date'].min()
                        last_withdrawal_date = outgoing_txns_valid_dates['trx_date'].max()

                        if last_withdrawal_date is not pd.NaT and first_deposit_date is not pd.NaT and last_withdrawal_date > first_deposit_date:
                            time_diff = (last_withdrawal_date - first_deposit_date).days
                            if time_diff <= self.mule_withdrawal_time_window_days:
                                count += 1
                                details['quick_withdrawal_time_diff_days'] = time_diff
                                details['first_deposit_date'] = str(first_deposit_date)
                                details['last_withdrawal_date'] = str(last_withdrawal_date)

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if not entity_wires.empty and entity_type == 'Particulier':
                if len(entity_wires) > self.mule_individual_wire_threshold:
                    count += 1
                    details['individual_wire_count'] = len(entity_wires)
                    details['wires'] = entity_wires[['wire_key', 'amount', 'wire_date']].to_dict(orient='records')


        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class FrequentEmailWireTransfersRule(DetectionRule):
    """
    Règle pour détecter l'utilisation fréquente de transferts par courriel et de virements internationaux.
    """
    def __init__(self, config: Dict):
        super().__init__('frequent_email_wire_transfers', config)
        # Aucun seuil spécifique nécessaire à partir de la configuration pour cette règle, juste les compteurs.

    def get_description(self) -> str:
        return "Utilisation fréquente de transferts par courriel et de virements internationaux"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}

        if transactions_df is not None and not transactions_df.empty:
            email_transfers = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['transaction_type_desc'] == 'Transfert Internet')
            ].copy()
            if not email_transfers.empty:
                 count += len(email_transfers)
                 details['email_transfers'] = email_transfers[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()
            if not entity_wires.empty:
                count += len(entity_wires)
                details['wires'] = entity_wires[['wire_key', 'amount', 'wire_date']].to_dict(orient='records')

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class MixedFundsBetweenAccountsRule(DetectionRule):
    """
    Règle pour détecter le mélange de fonds entre différents comptes personnels et professionnels.
    """
    def __init__(self, config: Dict):
        super().__init__('mixed_funds_between_accounts', config)
        # Aucun seuil spécifique nécessaire à partir de la configuration pour cette règle, juste la logique.

    def get_description(self) -> str:
        return "Mélange de fonds entre comptes personnels et professionnels"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]
                details['entity_type'] = entity_type

        # Vérifier les transactions pour les types de compte mixtes
        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            if not entity_txns.empty and 'account_type_desc' in entity_txns.columns:
                account_types_in_txns = entity_txns['account_type_desc'].dropna().unique()

                if len(account_types_in_txns) > 1:
                    count += 1
                    details['transaction_account_types'] = account_types_in_txns.tolist()

                    if entity_type == 'Particulier' and 'Entreprise' in account_types_in_txns:
                         count += 2 # Score plus élevé pour le mélange personnel et professionnel si l'entité est un particulier
                         details['personal_business_mix_highlight'] = True

        # Vérifier les virements (la logique originale vérifiait les pays uniques dans les virements pour les particuliers)
        # Cette partie de la logique semble légèrement distincte mais était sous le même nom de règle.
        # La conserver ici pour l'instant, mais elle pourrait justifier sa propre règle si la complexité augmente.
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if not entity_wires.empty:
                # Vérifier les pays uniques impliqués dans les virements pour les particuliers
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()

                # Les seuils du code original étaient > 5 (score +2) et > 3 (score +1)
                # Définir les paramètres pour ces seuils, en supposant qu'ils peuvent être configurés.
                multi_country_threshold_high = self.config['detection_params'].get('mixed_funds_multi_country_threshold_high', 5)
                multi_country_threshold_low = self.config['detection_params'].get('mixed_funds_multi_country_threshold_low', 3)
                multi_country_score_high = self.config['rules'].get('mixed_funds_between_accounts', {}).get('scores', [0,0,0])[-1] if len(self.rule_config.get('scores',[])) >= 3 else 2 # Supposons le score le plus élevé pour > seuil élevé
                multi_country_score_low = self.config['rules'].get('mixed_funds_between_accounts', {}).get('scores', [0,0,0])[-2] if len(self.rule_config.get('scores',[])) >= 2 else 1 # Supposons le deuxième score le plus élevé pour > seuil bas

                if unique_countries > multi_country_threshold_high:
                    count += 2 # Ajout direct au compteur basé sur la logique originale
                    details['unique_wire_countries'] = unique_countries
                    details['multi_country_level'] = 'high'
                elif unique_countries > multi_country_threshold_low:
                    count += 1 # Ajout direct au compteur basé sur la logique originale
                    details['unique_wire_countries'] = unique_countries
                    details['multi_country_level'] = 'low'

        # Retourner le score basé sur le compteur agrégé
        # Note : La logique originale ajoutait directement des scores basés sur des compteurs au sein de la méthode.
        # La classe de base _get_graduated_score est conçue pour des seuils menant à un score unique.
        # La logique de cette règle combine plusieurs compteurs et conditions pour un seul compteur total,
        # qui est ensuite passé à _get_graduated_score. Cela pourrait nécessiter un ajustement si
        # la structure de notation doit être plus granulaire en fonction de la partie spécifique de la règle qui se déclenche.
        score = self._get_graduated_score(count)
        return score, details

class HighVolumeDepositsRule(DetectionRule):
    """
    Règle pour détecter un volume inhabituellement élevé de dépôts.
    """
    def __init__(self, config: Dict):
        super().__init__('high_volume_deposits', config)
        # La logique de règle originale comptait simplement le nombre de dépôts.
        # Les seuils et scores dans la configuration détermineront le score basé sur ce compteur.

    def get_description(self) -> str:
        return "Volume inhabituellement élevé de dépôts"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}

        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            count = len(deposits)

            if count > 0:
                 details['deposit_count'] = count
                 # Inclure les détails des dépôts, limités à un nombre raisonnable si nombreux
                 if len(deposits) > 10: # Limite arbitraire pour les détails afin d'éviter une sortie excessive
                      details['sample_deposits'] = deposits.head(10)[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')
                 else:
                      details['all_deposits'] = deposits[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class StructuredDepositsRule(DetectionRule):
    """
    Règle pour détecter les dépôts structurés juste en dessous du seuil de déclaration.
    """
    def __init__(self, config: Dict):
        super().__init__('structured_deposits_below_threshold', config)
        # Extraire les paramètres spécifiques à la règle de la configuration
        self.threshold_amount = self.config['detection_params'].get('threshold_amount', 10000)
        self.margin = self.config['detection_params'].get('margin', 1000)
        self.lower_bound = self.threshold_amount - self.margin

    def get_description(self) -> str:
        return "Dépôts structurés sous le seuil de déclaration de 10 000$"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}
        structured_deposits_list = []

        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts pour cette entité
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            if not deposits.empty:
                # Identifier les dépôts structurés sous le seuil dans la marge
                structured_deposits = deposits[
                    (deposits['amount'] >= self.lower_bound) &
                    (deposits['amount'] < self.threshold_amount)
                ].copy()

                count = len(structured_deposits)

                if not structured_deposits.empty:
                    # Inclure les détails sur les dépôts structurés
                    structured_deposits_list = structured_deposits[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records')
                    details['structured_deposits'] = structured_deposits_list

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class QuickWithdrawalsRule(DetectionRule):
    """
    Règle pour détecter les retraits rapides après les dépôts.
    """
    def __init__(self, config: Dict):
        super().__init__('quick_withdrawals_after_deposits', config)
        # Extraire les paramètres spécifiques à la règle de la configuration
        self.withdrawal_time_window_days = self.config['detection_params'].get('quick_withdrawal_time_window_days', 3)
        self.withdrawal_percentage_threshold = self.config['detection_params'].get('quick_withdrawal_percentage_threshold', 0.7) # 70%

    def get_description(self) -> str:
        return "Retraits rapides après les dépôts"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}
        triggered_pairs = []

        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            if not entity_txns.empty:
                entity_txns = self._convert_dates(entity_txns, 'trx_date')
                # Supprimer les transactions avec des dates NaT après conversion
                entity_txns = entity_txns.dropna(subset=['trx_date'])

                if not entity_txns.empty:
                    deposits = entity_txns[entity_txns['sign'] == '+'].sort_values('trx_date').reset_index(drop=True)
                    withdrawals = entity_txns[entity_txns['sign'] == '-'].sort_values('trx_date').reset_index(drop=True)

                    if not deposits.empty and not withdrawals.empty:
                        for i in range(len(deposits)):
                            deposit = deposits.iloc[i]
                            deposit_date = deposit['trx_date']
                            deposit_amount = deposit['amount']

                            # Trouver les retraits ultérieurs dans la fenêtre temporelle
                            time_window_end = deposit_date + pd.Timedelta(days=self.withdrawal_time_window_days)
                            subsequent_withdrawals = withdrawals[
                                (withdrawals['trx_date'] > deposit_date) &
                                (withdrawals['trx_date'] <= time_window_end)
                            ].copy()

                            if not subsequent_withdrawals.empty:
                                withdrawal_amount_sum = subsequent_withdrawals['amount'].sum()

                                if withdrawal_amount_sum >= self.withdrawal_percentage_threshold * deposit_amount:
                                    count += 1
                                    # Ajouter les détails pour la paire déclenchée (dépôt et retraits ultérieurs)
                                    triggered_pairs.append({
                                        'deposit': deposit[['transaction_key', 'amount', 'trx_date']].to_dict(),
                                        'subsequent_withdrawals': subsequent_withdrawals[['transaction_key', 'amount', 'trx_date']].to_dict(orient='records'),
                                        'withdrawal_sum': withdrawal_amount_sum,
                                        'withdrawal_percentage_of_deposit': round((withdrawal_amount_sum / deposit_amount) * 100, 2) if deposit_amount > 0 else 0
                                    })
        if triggered_pairs:
             details['triggered_deposit_withdrawal_pairs'] = triggered_pairs

        # Retourner le score basé sur le compteur agrégé (nombre de paires déclenchées)
        score = self._get_graduated_score(count)
        return score, details

class ForeignExchangeWiresRule(DetectionRule):
    """
    Règle pour détecter les gros virements provenant de sociétés de change.
    """
    def __init__(self, config: Dict):
        super().__init__('foreign_exchange_wires', config)
        # Aucun seuil spécifique nécessaire à partir de la configuration pour le compteur, juste la liste/modèle pour les initiateurs
        self.forex_keywords = self.config['detection_params'].get('forex_keywords', 'change|forex|exchange|money')

    def get_description(self) -> str:
        return "Gros virements provenant de sociétés de change"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0
        details = {}
        triggered_wires_list = []

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if not entity_wires.empty:
                # Filtrer les virements dont l'initiateur contient des mots-clés liés au change
                forex_wires = entity_wires[
                    entity_wires['originator'].str.contains(self.forex_keywords, case=False, na=False)
                ].copy() # Ajout de copy pour éviter SettingWithCopyWarning

                count = len(forex_wires)

                if not forex_wires.empty:
                    # Inclure les détails sur les virements déclenchés
                    triggered_wires_list = forex_wires[['wire_key', 'originator', 'amount', 'wire_date']].to_dict(orient='records')
                    details['triggered_forex_wires'] = triggered_wires_list

        # Retourner le score basé sur le compteur agrégé
        score = self._get_graduated_score(count)
        return score, details

class InconsistentActivityRule(DetectionRule):
    """
    Règle pour détecter une activité incohérente avec le profil ou le type d'entreprise.
    """
    def __init__(self, config: Dict):
        super().__init__('inconsistent_activity', config)
        # Extraire les paramètres spécifiques à la règle de la configuration
        # Seuils pour les pays uniques dans les virements pour les particuliers
        self.multi_country_threshold_high = self.config['detection_params'].get('inconsistent_activity_multi_country_threshold_high', 5)
        self.multi_country_threshold_low = self.config['detection_params'].get('inconsistent_activity_multi_country_threshold_low', 3)
        # Supposons que la notation est gérée par _get_graduated_score basée sur le compteur agrégé

    def get_description(self) -> str:
        return "Activité incohérente avec le profil ou le type d'entreprise"

    def apply(self, entity_id: Any, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None) -> Tuple[int, Dict]:
        count = 0 # Ce compteur agrègera les déclencheurs au sein de cette règle
        details = {}

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]
                details['entity_type'] = entity_type

        # Vérifier les transactions pour l'incohérence basée sur le type d'entité
        if transactions_df is not None and not transactions_df.empty and entity_type is not None:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            if not entity_txns.empty:
                if entity_type == 'Particulier':
                    # Rechercher les types de transaction typiquement associés aux entreprises
                    commercial_keywords = 'Entreprise|Commercial|Business'
                    commercial_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains(commercial_keywords, case=False, na=False)
                    ].copy()

                    if not commercial_txns.empty:
                        count += len(commercial_txns)
                        details['commercial_txns_in_individual'] = commercial_txns[['transaction_key', 'transaction_type_desc', 'amount', 'trx_date']].to_dict(orient='records')

                elif entity_type == 'Entreprise':
                    # Rechercher les types de transaction typiquement associés aux particuliers
                    personal_keywords = 'Personnel|Personal'
                    personal_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains(personal_keywords, case=False, na=False)
                    ].copy()

                    if not personal_txns.empty:
                        count += len(personal_txns)
                        details['personal_txns_in_business'] = personal_txns[['transaction_key', 'transaction_type_desc', 'amount', 'trx_date']].to_dict(orient='records')

        # Vérifier les virements pour les pays uniques si l'entité est un particulier
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if not entity_wires.empty:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()

                # Ajouter au compteur en fonction des seuils de pays uniques
                if unique_countries > self.multi_country_threshold_high:
                    count += 2 # Ajouter 2 au compteur si supérieur au seuil élevé
                    details['unique_wire_countries'] = unique_countries
                    details['multi_country_level'] = 'high'
                elif unique_countries > self.multi_country_threshold_low:
                    count += 1 # Ajouter 1 au compteur si supérieur au seuil bas
                    details['unique_wire_countries'] = unique_countries
                    details['multi_country_level'] = 'low'

        # Retourner le score basé sur le compteur agrégé de toutes les vérifications au sein de cette règle
        score = self._get_graduated_score(count)
        return score, details

# Add other rule classes here following the same pattern...
# For example:
# class SplitCashDepositsRule(DetectionRule): ...
# class SuspectedMoneyMulesRule(DetectionRule): ... 