# aml_detection/src_fr/rules.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import pandas as pd

@dataclass
class BaseRule:
    """Classe de base pour toutes les règles de détection."""
    name: str
    description: str
    # Ceci contiendra les seuils et les scores pour cette règle spécifique
    config: Dict[str, Any] = field(default_factory=dict)

    def _get_graduated_score(self, count: int) -> int:
        """
        Obtenir le score approprié basé sur le compteur et les seuils de la règle.

        Args:
            count (int): Nombre d'instances détectées

        Returns:
            int: Score basé sur les seuils
        """
        if not self.config or 'thresholds' not in self.config or 'scores' not in self.config:
            return 0 # Should not happen if config is loaded correctly

        thresholds = self.config['thresholds']
        scores = self.config['scores']

        # Trouver le seuil le plus élevé que le compteur dépasse
        for i in range(len(thresholds) - 1, -1, -1):
            if count >= thresholds[i]:
                return scores[i]

        return 0

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        """
        Exécute la logique de détection spécifique à la règle.
        Doit être implémenté par les sous-classes.

        Args:
            entity_id (str): L'ID de l'entité évaluée.
            transactions_df (pd.DataFrame | None): DataFrame contenant les données de transaction.
            wires_df (pd.DataFrame | None): DataFrame contenant les données de virement bancaire.
            entity_df (pd.DataFrame | None): DataFrame contenant les données d'entité.
            parent_helper (Any): Une référence à l'instance parente RuleBasedDetection pour les méthodes d'aide (comme la conversion de dates).

        Returns:
            int: Le compte des instances détectées pour cette règle.
        """
        raise NotImplementedError("La méthode 'detect' doit être implémentée par les sous-classes")

    def get_score(self, count: int) -> int:
        """
        Calcule le score de la règle basé sur le compte détecté.
        """
        return self._get_graduated_score(count)


# Méthode d'aide (déplacée de RuleBasedDetection pour une utilisation partagée si nécessaire)
# Alternativement, cela pourrait rester dans la classe principale et être passé via parent_helper
def _convert_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Méthode auxiliaire pour convertir les colonnes de dates en toute sécurité.

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne de date
        date_column (str): Nom de la colonne contenant les dates

    Returns:
        pd.DataFrame: DataFrame avec les dates converties
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    try:
        # Essayer plusieurs formats de date
        df[date_column] = pd.to_datetime(df[date_column], format='%d%b%Y', errors='coerce')
        # Si toutes les dates sont NaT, essayer sans format spécifique
        if df[date_column].isna().all():
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    except Exception:
        # Si la conversion échoue, essayer sans format spécifique
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df


# --- Implémentations de Règles Spécifiques ---

@dataclass
class LargeWireTransfersFollowedByOutgoingRule(BaseRule):
    def __post_init__(self):
         # Assurer que la description est définie si non fournie
        if not self.description:
            self.description = "Reçoit soudainement des virements importants suivis de transferts sortants"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements entrants pour cette entité
            incoming_wires = wires_df[(wires_df['party_key'] == entity_id) & (wires_df['sign'] == '+')].copy()

            if len(incoming_wires) > 0:
                # Identifier les gros virements (plus de 5000$) - Utilisation d'un seuil codé en dur pour l'instant, pourrait être dans la config
                large_wires = incoming_wires[incoming_wires['amount'] > 5000]

                if len(large_wires) > 0:
                    count += 1

                    # Vérifier les transactions sortantes après les dates des gros virements
                    if transactions_df is not None and not transactions_df.empty:
                        # Utiliser la fonction d'aide partagée pour la conversion de dates
                        large_wires = _convert_dates(large_wires, 'wire_date')

                        if not large_wires.empty and 'wire_date' in large_wires.columns:
                            # Assurer que la conversion de date a réussi
                             if not large_wires['wire_date'].isna().all():
                                min_date = large_wires['wire_date'].min()

                                transactions_df_converted = _convert_dates(transactions_df, 'trx_date') # Convertir aussi les dates de transaction

                                transactions_txns_converted = transactions_df_converted # Correction: Renommer pour cohérence avec l'original
                                outgoing_txns = transactions_txns_converted[
                                    (transactions_txns_converted['party_key'] == entity_id) &
                                    (transactions_txns_converted['sign'] == '-') &
                                    (transactions_txns_converted['trx_date'] > min_date)
                                ].copy()

                                if len(outgoing_txns) > 3:
                                    count += 1

                                    unique_types = outgoing_txns['transaction_type_desc'].nunique()
                                    if unique_types > 1:
                                        count += 1

        # Vérifier aussi les gros dépôts en espèces
        if transactions_df is not None and not transactions_df.empty:
            transactions_df_converted = _convert_dates(transactions_df, 'trx_date')

            cash_deposits = transactions_df_converted[
                (transactions_df_converted['party_key'] == entity_id) &
                (transactions_df_converted['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df_converted['sign'] == '+')
            ].copy()

            if len(cash_deposits) > 0:
                # Identifier les gros dépôts (plus de 5000$) - Codé en dur pour l'instant
                large_deposits = cash_deposits[cash_deposits['amount'] > 5000]
                if len(large_deposits) > 0:
                    count += 1

                if not large_deposits.empty and 'trx_date' in large_deposits.columns:
                     # Assurer que la conversion de date a réussi
                    if not large_deposits['trx_date'].isna().all():
                        min_date = large_deposits['trx_date'].min()

                        outgoing_txns = transactions_df_converted[
                            (transactions_df_converted['party_key'] == entity_id) &
                            (transactions_df_converted['sign'] == '-') &
                            (transactions_df_converted['trx_date'] > min_date)
                        ].copy()

                        if len(outgoing_txns) > 3:
                            count += 1

        return count

@dataclass
class SanctionedCountriesTransactionsRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Transactions avec des pays sanctionnés ou à haut risque"
        # Supposons que suspicious_countries peut provenir de la configuration principale
        self.suspicious_countries = self.config.get('suspicious_countries', ['CN', 'HK', 'AE', 'IR', 'KW']) # Liste par défaut si non dans la config de la règle

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if len(entity_wires) > 0:
                # Vérifier les pays suspects dans les codes pays de l'initiateur ou du bénéficiaire
                # Supposons que 'CN', 'HK', 'AE', 'IR', 'KW' sont des codes pays, pas des noms complets
                # Utiliser .str.upper() pour une correspondance insensible à la casse si les codes pays sont de casse mixte
                suspicious_origin = entity_wires[entity_wires['originator_country'].isin(self.suspicious_countries)]
                suspicious_dest = entity_wires[entity_wires['beneficiary_country'].isin(self.suspicious_countries)]

                count = len(suspicious_origin) + len(suspicious_dest)

        return count

@dataclass
class SplitCashDepositsSameDayRule(BaseRule):
    def __post_init__(self):
        if not self.description:
             self.description = "Dépôts en espèces fractionnés le même jour à plusieurs endroits"
        # Supposons que threshold_amount peut provenir de la configuration principale
        self.threshold_amount = self.config.get('threshold_amount', 10000) # Seuil par défaut si non dans la config de la règle

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
            # Filtrer les dépôts en espèces
            cash_deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['transaction_type_desc'] == 'Depot Especes') &
                (transactions_df['sign'] == '+')
            ].copy()

            if len(cash_deposits) > 0:
                # Convertir les dates en utilisant la fonction d'aide
                cash_deposits = _convert_dates(cash_deposits, 'trx_date')

                # Supprimer les lignes où la conversion de date a échoué
                cash_deposits = cash_deposits.dropna(subset=['trx_date'])

                if not cash_deposits.empty:
                    # Convertir datetime en chaîne de date pour le groupement
                    cash_deposits['date_str'] = cash_deposits['trx_date'].dt.strftime('%Y-%m-%d')

                    # Grouper par chaîne de date
                    # Vérifier si la colonne 'branch' existe avant de grouper
                    if 'branch' in cash_deposits.columns:
                         daily_deposits = cash_deposits.groupby('date_str').agg(
                             branch_nunique=('branch', 'nunique'),
                             amount_sum=('amount', 'sum')
                         ).reset_index()

                         # Trouver les jours suspects (plus d'une succursale ET montant total > seuil)
                         # Utilisation du seuil_amount configuré
                         suspicious_days = daily_deposits[
                             (daily_deposits['branch_nunique'] > 1) &
                             (daily_deposits['amount_sum'] > self.threshold_amount)
                         ]

                         count = len(suspicious_days)
                    else:
                        # Si la colonne 'branch' n'existe pas, la logique de la règle ne peut pas être appliquée
                        # Optionnellement, enregistrer un avertissement ou retourner 0
                        pass # La règle ne peut pas être appliquée sans l'information de 'branch'

        return count

@dataclass
class SuspectedMoneyMulesRule(BaseRule):
    def __post_init__(self):
         # Assurer que la description est définie si non fournie
         if not self.description:
            self.description = "Utilisation de mules financières suspectées"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]

        if transactions_df is not None and not transactions_df.empty:
            incoming_txns = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            if len(incoming_txns) > 0:
                # Vérifier si 'transaction_type_desc' existe avant de filtrer
                if 'transaction_type_desc' in incoming_txns.columns:
                    deposit_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Depot Especes'])
                    email_transfer_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Transfert Internet'])

                    # Seuils codés en dur pour l'instant, pourraient être dans la config
                    if entity_type == 'Particulier' and (deposit_count > 5 or email_transfer_count > 5):
                        count += 1

                outgoing_txns = transactions_df[
                    (transactions_df['party_key'] == entity_id) &
                    (transactions_df['sign'] == '-')
                ].copy()

                if len(outgoing_txns) > 0 and len(incoming_txns) > 0:
                    # Utiliser la fonction d'aide partagée pour la conversion de dates
                    incoming_txns = _convert_dates(incoming_txns, 'trx_date')
                    outgoing_txns = _convert_dates(outgoing_txns, 'trx_date')

                    # Assurer que la conversion de date a réussi et que les dates existent
                    if not incoming_txns['trx_date'].empty and not incoming_txns['trx_date'].isna().all() and \
                       not outgoing_txns['trx_date'].empty and not outgoing_txns['trx_date'].isna().all():

                        # Regarder le premier dépôt et le dernier retrait
                        first_deposit_date = incoming_txns['trx_date'].min()
                        last_withdrawal_date = outgoing_txns['trx_date'].max()

                        if last_withdrawal_date > first_deposit_date:
                            # Vérifier la différence de temps en jours (3 jours codés en dur)
                            time_diff = (last_withdrawal_date - first_deposit_date).days
                            if time_diff is not pd.NaT and time_diff <= 3:
                                count += 1


        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            # Seuil codé en dur pour l'instant, pourrait être dans la config
            if len(entity_wires) > 0 and entity_type == 'Particulier' and len(entity_wires) > 2:
                count += 1

        return count

@dataclass
class FrequentEmailWireTransfersRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Utilisation fréquente de transferts par courriel et de virements internationaux"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
             # Vérifier si 'transaction_type_desc' existe avant de filtrer
            if 'transaction_type_desc' in transactions_df.columns:
                email_transfers = transactions_df[
                    (transactions_df['party_key'] == entity_id) &
                    (transactions_df['transaction_type_desc'] == 'Transfert Internet')
                ].copy()
                count += len(email_transfers)

        if wires_df is not None and not wires_df.empty:
            wires = wires_df[wires_df['party_key'] == entity_id].copy()
            count += len(wires)

        return count

@dataclass
class MixedFundsBetweenAccountsRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Mélange de fonds entre comptes personnels et professionnels"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]

        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            # Vérifier si 'account_type_desc' existe avant de vérifier les valeurs uniques
            if len(entity_txns) > 0 and 'account_type_desc' in entity_txns.columns:
                account_types = entity_txns['account_type_desc'].dropna().unique()

                if len(account_types) > 1:
                    count += 1

                    # Vérifier le mélange spécifique
                    if entity_type == 'Particulier' and 'Entreprise' in account_types:
                        count += 2 # Compteur plus élevé pour ce mélange spécifique

        # Partie des virements bancaires - semble moins pertinente pour le mélange de comptes, sauf si les données de virement incluent les types de compte ?
        # Le code original vérifie les pays uniques dans les virements pour les types Particulier.
        # Cela semble incohérent avec le nom de la règle "Mixed Funds Between Accounts".
        # Je vais conserver la logique originale pour l'instant mais noter l'incohérence potentielle.
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

             # Vérifier si les colonnes de pays existent avant de concaténer
            if len(entity_wires) > 0 and 'originator_country' in entity_wires.columns and 'beneficiary_country' in entity_wires.columns:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()

                # Seuils codés en dur pour les pays uniques, pourraient être dans la config
                if unique_countries > 5:
                    count += 2
                elif unique_countries > 3:
                    count += 1

        return count

@dataclass
class HighVolumeDepositsRule(BaseRule):
    def __post_init__(self):
        if not self.description:
             self.description = "Volume inhabituellement élevé de dépôts"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            count = len(deposits) # Le compte est simplement le nombre de dépôts

        return count

@dataclass
class StructuredDepositsBelowThresholdRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Dépôts structurés sous le seuil de déclaration de 10 000$"
         # Supposons que threshold_amount et margin proviennent de la configuration principale
        self.threshold_amount = self.config.get('threshold_amount', 10000) # Seuil par défaut si non dans la config de la règle
        self.margin = self.config.get('margin', 1000) # Marge par défaut si non dans la config de la règle


    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            if len(deposits) > 0:
                # Vérifier si la colonne 'amount' existe avant de filtrer
                if 'amount' in deposits.columns:
                    # Filtrer les dépôts dans la plage de seuil
                    structured_deposits = deposits[
                        (deposits['amount'] >= self.threshold_amount - self.margin) &
                        (deposits['amount'] < self.threshold_amount)
                    ]

                    count = len(structured_deposits)

        return count

@dataclass
class QuickWithdrawalsAfterDepositsRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Retraits rapides après les dépôts"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            if len(entity_txns) > 0:
                # Utiliser la fonction d'aide partagée pour la conversion de dates
                entity_txns = _convert_dates(entity_txns, 'trx_date')

                # Supprimer les lignes où la conversion de date a échoué
                entity_txns = entity_txns.dropna(subset=['trx_date'])

                if not entity_txns.empty:
                    deposits = entity_txns[entity_txns['sign'] == '+'].sort_values('trx_date')
                    withdrawals = entity_txns[entity_txns['sign'] == '-'].sort_values('trx_date')

                    if not deposits.empty and not withdrawals.empty:
                        # Vérifier si les colonnes 'amount' et 'trx_date' existent avant d'itérer
                        if 'amount' in deposits.columns and 'trx_date' in deposits.columns and \
                           'amount' in withdrawals.columns and 'trx_date' in withdrawals.columns:
                            for _, deposit in deposits.iterrows():
                                deposit_date = deposit['trx_date']
                                deposit_amount = deposit['amount']

                                # Trouver les retraits dans les 3 jours suivant le dépôt
                                subsequent_withdrawals = withdrawals[
                                    (withdrawals['trx_date'] > deposit_date) &
                                    (withdrawals['trx_date'] <= deposit_date + pd.Timedelta(days=3)) # 3 jours codés en dur
                                ]

                                if not subsequent_withdrawals.empty:
                                    withdrawal_amount = subsequent_withdrawals['amount'].sum()

                                    # Vérifier si le montant du retrait est au moins 70% du montant du dépôt
                                    if withdrawal_amount >= 0.7 * deposit_amount: # 70% codé en dur
                                        count += 1 # Compter chaque dépôt suivi de retraits qualifiants

        return count

@dataclass
class ForeignExchangeWiresRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Gros virements provenant de sociétés de change"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            # Vérifier si la colonne 'originator' existe avant de rechercher
            if len(entity_wires) > 0 and 'originator' in entity_wires.columns:
                # Rechercher des mots-clés dans le nom de l'initiateur (insensible à la casse)
                # Mots-clés codés en dur, pourraient être dans la config
                forex_wires = entity_wires[
                    entity_wires['originator'].str.contains('change|forex|exchange|money', case=False, na=False)
                ]

                count = len(forex_wires) # Compter le nombre de tels virements

        return count

@dataclass
class InconsistentActivityRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Activité incohérente avec le profil ou le type d'entreprise"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        entity_type = None
        if entity_df is not None and not entity_df.empty:
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty and 'account_type_desc' in entity_info.columns:
                entity_type = entity_info['account_type_desc'].iloc[0]

        if transactions_df is not None and not transactions_df.empty and entity_type is not None:
            entity_txns = transactions_df[transactions_df['party_key'] == entity_id].copy()

            # Vérifier si 'transaction_type_desc' existe avant de rechercher
            if len(entity_txns) > 0 and 'transaction_type_desc' in entity_txns.columns:
                if entity_type == 'Particulier':
                    # Rechercher les types de transaction typiquement associés aux entreprises
                    commercial_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Entreprise|Commercial|Business', case=False, na=False) # Mots-clés codés en dur
                    ]
                    count += len(commercial_txns)

                elif entity_type == 'Entreprise':
                    # Rechercher les types de transaction typiquement associés aux particuliers
                    personal_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Personnel|Personal', case=False, na=False) # Mots-clés codés en dur
                    ]
                    count += len(personal_txns)

        # Partie des virements bancaires - vérifie les pays uniques pour les particuliers
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

             # Vérifier si les colonnes de pays existent avant de concaténer
            if len(entity_wires) > 0 and 'originator_country' in entity_wires.columns and 'beneficiary_country' in entity_wires.columns:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()

                # Seuils codés en dur pour les pays uniques
                if unique_countries > 5:
                    count += 2
                elif unique_countries > 3:
                    count += 1

        return count

@dataclass
class PriorSuspiciousFlagBoostRule(BaseRule):
    # Cette règle ne détecte pas un modèle, elle fournit juste un boost si un drapeau est défini
    # Sa méthode 'detect' vérifiera simplement le drapeau et retournera un compte (0 ou 1)
    def __post_init__(self):
        if not self.description:
             self.description = "Compte précédemment signalé comme suspect"
        # Supposons que prior_suspicious_flag_boost provienne de la configuration principale
        self.boost_amount = self.config.get('prior_suspicious_flag_boost', 20) # Boost par défaut

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        """
        Détecte si le drapeau prior_suspicious_flag est défini pour l'entité.
        Retourne 1 si défini, 0 sinon.
        """
        if entity_df is not None and not entity_df.empty:
            if 'prior_suspicious_flag' in entity_df.columns:
                entity_info = entity_df[entity_df['party_key'] == entity_id]
                if not entity_info.empty and entity_info['prior_suspicious_flag'].iloc[0] == 1:
                    return 1 # Détecté (le drapeau est défini)
        return 0 # Non détecté

    def get_score(self, count: int) -> int:
        """
        Retourne le montant du boost si le compte est de 1, 0 sinon.
        """
        if count > 0: # Si le drapeau a été détecté (count == 1)
            # Cette règle n'utilise pas de notation graduée basée sur des seuils.
            # Elle fournit un montant de boost fixe si elle est déclenchée.
            return self.boost_amount
        return 0
