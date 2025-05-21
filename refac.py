# aml_detection/src_fr/rules.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import pandas as pd

@dataclass
class BaseRule:
    """Base dataclass for all detection rules."""
    name: str
    description: str
    # This will hold the thresholds and scores for this specific rule
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
        Performs the specific detection logic for the rule.
        Must be implemented by subclasses.

        Args:
            entity_id (str): The ID of the entity being evaluated.
            transactions_df (pd.DataFrame | None): DataFrame containing transaction data.
            wires_df (pd.DataFrame | None): DataFrame containing wire transfer data.
            entity_df (pd.DataFrame | None): DataFrame containing entity data.
            parent_helper (Any): A reference to the parent RuleBasedDetection instance for helper methods (like date conversion).

        Returns:
            int: The count of detected instances for this rule.
        """
        raise NotImplementedError("Subclasses must implement the detect method")

    def get_score(self, count: int) -> int:
        """
        Calculates the score for the rule based on the detected count.
        """
        return self._get_graduated_score(count)


# Helper method (moved from RuleBasedDetection for shared use if needed)
# Alternatively, this could stay in the main class and be passed via parent_helper
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
        # Try multiple date formats
        df[date_column] = pd.to_datetime(df[date_column], format='%d%b%Y', errors='coerce')
        if df[date_column].isna().all():
            # If all dates are NaT, try without specific format
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    except Exception:
        # If conversion fails, try without specific format
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df


# --- Specific Rule Implementations ---

@dataclass
class LargeWireTransfersFollowedByOutgoingRule(BaseRule):
    def __post_init__(self):
         # Ensure description is set if not provided
        if not self.description:
            self.description = "Reçoit soudainement des virements importants suivis de transferts sortants"

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if wires_df is not None and not wires_df.empty:
            # Filtrer les virements entrants pour cette entité
            incoming_wires = wires_df[(wires_df['party_key'] == entity_id) & (wires_df['sign'] == '+')].copy()

            if len(incoming_wires) > 0:
                # Identifier les gros virements (plus de 5000$) - Using a hardcoded threshold for now, could be in config
                large_wires = incoming_wires[incoming_wires['amount'] > 5000]

                if len(large_wires) > 0:
                    count += 1

                    # Vérifier les transactions sortantes après les dates des gros virements
                    if transactions_df is not None and not transactions_df.empty:
                        # Use the shared helper function for date conversion
                        large_wires = _convert_dates(large_wires, 'wire_date')

                        if not large_wires.empty and 'wire_date' in large_wires.columns:
                            # Ensure the date conversion was successful
                             if not large_wires['wire_date'].isna().all():
                                min_date = large_wires['wire_date'].min()

                                transactions_df_converted = _convert_dates(transactions_df, 'trx_date') # Convert transactions dates too

                                outgoing_txns = transactions_df_converted[
                                    (transactions_df_converted['party_key'] == entity_id) &
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
                # Identify large deposits (more than 5000$) - Hardcoded for now
                large_deposits = cash_deposits[cash_deposits['amount'] > 5000]
                if len(large_deposits) > 0:
                    count += 1

                if not large_deposits.empty and 'trx_date' in large_deposits.columns:
                     # Ensure the date conversion was successful
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
        # Assuming suspicious_countries might come from the main config
        self.suspicious_countries = self.config.get('suspicious_countries', ['CN', 'HK', 'AE', 'IR', 'KW']) # Default list if not in rule config

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if len(entity_wires) > 0:
                # Check for suspicious countries in originator or beneficiary country codes
                # Assuming 'CN', 'HK', 'AE', 'IR', 'KW' are country codes, not full names
                # Use .str.upper() for case-insensitive matching if country codes are mixed case
                suspicious_origin = entity_wires[entity_wires['originator_country'].isin(self.suspicious_countries)]
                suspicious_dest = entity_wires[entity_wires['beneficiary_country'].isin(self.suspicious_countries)]

                count = len(suspicious_origin) + len(suspicious_dest)

        return count

@dataclass
class SplitCashDepositsSameDayRule(BaseRule):
    def __post_init__(self):
        if not self.description:
             self.description = "Dépôts en espèces fractionnés le même jour à plusieurs endroits"
        # Assuming threshold_amount might come from the main config
        self.threshold_amount = self.config.get('threshold_amount', 10000) # Default threshold if not in rule config

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
                # Convertir les dates using the helper
                cash_deposits = _convert_dates(cash_deposits, 'trx_date')

                # Supprimer les lignes où la conversion de date a échoué
                cash_deposits = cash_deposits.dropna(subset=['trx_date'])

                if not cash_deposits.empty:
                    # Convert datetime to date string for grouping
                    cash_deposits['date_str'] = cash_deposits['trx_date'].dt.strftime('%Y-%m-%d')

                    # Grouper par chaîne de date
                    # Check if 'branch' column exists before grouping
                    if 'branch' in cash_deposits.columns:
                         daily_deposits = cash_deposits.groupby('date_str').agg(
                             branch_nunique=('branch', 'nunique'),
                             amount_sum=('amount', 'sum')
                         ).reset_index()

                         # Trouver les jours suspects (more than 1 branch AND total amount > threshold)
                         # Using the configured threshold_amount
                         suspicious_days = daily_deposits[
                             (daily_deposits['branch_nunique'] > 1) &
                             (daily_deposits['amount_sum'] > self.threshold_amount)
                         ]

                         count = len(suspicious_days)
                    else:
                        # If 'branch' column doesn't exist, we cannot apply the rule logic
                        # Optionally log a warning or return 0
                        pass # Rule cannot be applied without 'branch' info

        return count

@dataclass
class SuspectedMoneyMulesRule(BaseRule):
    def __post_init__(self):
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
                # Check if 'transaction_type_desc' exists before filtering
                if 'transaction_type_desc' in incoming_txns.columns:
                    deposit_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Depot Especes'])
                    email_transfer_count = len(incoming_txns[incoming_txns['transaction_type_desc'] == 'Transfert Internet'])

                    # Hardcoded thresholds for now, could be in config
                    if entity_type == 'Particulier' and (deposit_count > 5 or email_transfer_count > 5):
                        count += 1

                outgoing_txns = transactions_df[
                    (transactions_df['party_key'] == entity_id) &
                    (transactions_df['sign'] == '-')
                ].copy()

                if len(outgoing_txns) > 0 and len(incoming_txns) > 0:
                    # Use the shared helper function for date conversion
                    incoming_txns = _convert_dates(incoming_txns, 'trx_date')
                    outgoing_txns = _convert_dates(outgoing_txns, 'trx_date')

                    # Ensure date conversion was successful and dates exist
                    if not incoming_txns['trx_date'].empty and not incoming_txns['trx_date'].isna().all() and \
                       not outgoing_txns['trx_date'].empty and not outgoing_txns['trx_date'].isna().all():

                        # Look at the first deposit and the last withdrawal
                        first_deposit_date = incoming_txns['trx_date'].min()
                        last_withdrawal_date = outgoing_txns['trx_date'].max()

                        if last_withdrawal_date > first_deposit_date:
                            # Check time difference in days (hardcoded 3 days)
                            time_diff = (last_withdrawal_date - first_deposit_date).days
                            if time_diff is not pd.NaT and time_diff <= 3:
                                count += 1


        if wires_df is not None and not wires_df.empty:
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            # Hardcoded threshold for now, could be in config
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
             # Check if 'transaction_type_desc' exists before filtering
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

            # Check if 'account_type_desc' exists before checking unique values
            if len(entity_txns) > 0 and 'account_type_desc' in entity_txns.columns:
                account_types = entity_txns['account_type_desc'].dropna().unique()

                if len(account_types) > 1:
                    count += 1

                    # Check for specific mix
                    if entity_type == 'Particulier' and 'Entreprise' in account_types:
                        count += 2 # Higher count for this specific mix

        # Wire transfers part - seems less relevant to account mixing unless wire data includes account types?
        # The original code checks for unique countries in wires for Particulier type.
        # This seems inconsistent with the rule name "Mixed Funds Between Accounts".
        # I will keep the original logic for now but note the potential inconsistency.
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

            if len(entity_wires) > 0:
                 # Check if country columns exist before concatenating
                 if 'originator_country' in entity_wires.columns and 'beneficiary_country' in entity_wires.columns:
                    unique_countries = pd.concat([
                        entity_wires['originator_country'].dropna(),
                        entity_wires['beneficiary_country'].dropna()
                    ]).nunique()

                    # Hardcoded thresholds for unique countries, could be in config
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

            count = len(deposits) # The count is simply the number of deposits

        return count

@dataclass
class StructuredDepositsBelowThresholdRule(BaseRule):
    def __post_init__(self):
        if not self.description:
            self.description = "Dépôts structurés sous le seuil de déclaration de 10 000$"
         # Assuming threshold_amount and margin come from the main config
        self.threshold_amount = self.config.get('threshold_amount', 10000) # Default threshold if not in rule config
        self.margin = self.config.get('margin', 1000) # Default margin if not in rule config


    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        count = 0

        if transactions_df is not None and not transactions_df.empty:
            deposits = transactions_df[
                (transactions_df['party_key'] == entity_id) &
                (transactions_df['sign'] == '+')
            ].copy()

            if len(deposits) > 0:
                # Check if 'amount' column exists before filtering
                if 'amount' in deposits.columns:
                    # Filter deposits within the threshold range
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
                # Use the shared helper function for date conversion
                entity_txns = _convert_dates(entity_txns, 'trx_date')

                # Remove rows where date conversion failed
                entity_txns = entity_txns.dropna(subset=['trx_date'])

                if not entity_txns.empty:
                    deposits = entity_txns[entity_txns['sign'] == '+'].sort_values('trx_date')
                    withdrawals = entity_txns[entity_txns['sign'] == '-'].sort_values('trx_date')

                    if not deposits.empty and not withdrawals.empty:
                        # Check if 'amount' and 'trx_date' columns exist before iterating
                        if 'amount' in deposits.columns and 'trx_date' in deposits.columns and \
                           'amount' in withdrawals.columns and 'trx_date' in withdrawals.columns:
                            for _, deposit in deposits.iterrows():
                                deposit_date = deposit['trx_date']
                                deposit_amount = deposit['amount']

                                # Find withdrawals within 3 days after the deposit
                                subsequent_withdrawals = withdrawals[
                                    (withdrawals['trx_date'] > deposit_date) &
                                    (withdrawals['trx_date'] <= deposit_date + pd.Timedelta(days=3)) # Hardcoded 3 days
                                ]

                                if not subsequent_withdrawals.empty:
                                    withdrawal_amount = subsequent_withdrawals['amount'].sum()

                                    # Check if withdrawal amount is at least 70% of deposit amount
                                    if withdrawal_amount >= 0.7 * deposit_amount: # Hardcoded 70%
                                        count += 1 # Count each deposit followed by qualifying withdrawals

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

            # Check if 'originator' column exists before searching
            if len(entity_wires) > 0 and 'originator' in entity_wires.columns:
                # Search for keywords in the originator name (case-insensitive)
                # Hardcoded keywords, could be in config
                forex_wires = entity_wires[
                    entity_wires['originator'].str.contains('change|forex|exchange|money', case=False, na=False)
                ]

                count = len(forex_wires) # Count the number of such wires

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

            # Check if 'transaction_type_desc' exists before searching
            if len(entity_txns) > 0 and 'transaction_type_desc' in entity_txns.columns:
                if entity_type == 'Particulier':
                    # Look for commercial transaction types for individuals
                    commercial_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Entreprise|Commercial|Business', case=False, na=False) # Hardcoded keywords
                    ]
                    count += len(commercial_txns)

                elif entity_type == 'Entreprise':
                    # Look for personal transaction types for businesses
                    personal_txns = entity_txns[
                        entity_txns['transaction_type_desc'].str.contains('Personnel|Personal', case=False, na=False) # Hardcoded keywords
                    ]
                    count += len(personal_txns)

        # Wire transfers part - checks for unique countries for individuals
        if wires_df is not None and not wires_df.empty and entity_type == 'Particulier':
            entity_wires = wires_df[wires_df['party_key'] == entity_id].copy()

             # Check if country columns exist before concatenating
            if len(entity_wires) > 0 and 'originator_country' in entity_wires.columns and 'beneficiary_country' in entity_wires.columns:
                unique_countries = pd.concat([
                    entity_wires['originator_country'].dropna(),
                    entity_wires['beneficiary_country'].dropna()
                ]).nunique()

                # Hardcoded thresholds for unique countries
                if unique_countries > 5:
                    count += 2
                elif unique_countries > 3:
                    count += 1

        return count

@dataclass
class PriorSuspiciousFlagBoostRule(BaseRule):
    # This rule doesn't detect a pattern, it just provides a boost if a flag is set
    # Its 'detect' method will just check the flag and return a count (0 or 1)
    def __post_init__(self):
        if not self.description:
             self.description = "Compte précédemment signalé comme suspect"
        # Assuming prior_suspicious_flag_boost comes from the main config
        self.boost_amount = self.config.get('prior_suspicious_flag_boost', 20) # Default boost

    def detect(self, entity_id: str, transactions_df: pd.DataFrame | None, wires_df: pd.DataFrame | None, entity_df: pd.DataFrame | None, parent_helper: Any) -> int:
        """
        Detects if the prior_suspicious_flag is set for the entity.
        Returns 1 if set, 0 otherwise.
        """
        if entity_df is not None and not entity_df.empty:
            if 'prior_suspicious_flag' in entity_df.columns:
                entity_info = entity_df[entity_df['party_key'] == entity_id]
                if not entity_info.empty and entity_info['prior_suspicious_flag'].iloc[0] == 1:
                    return 1 # Detected (flag is set)
        return 0 # Not detected

    def get_score(self, count: int) -> int:
        """
        Returns the boost amount if count is 1, 0 otherwise.
        """
        if count > 0: # If flag was detected (count == 1)
            # This rule doesn't use graduated scoring based on thresholds.
            # It provides a fixed boost amount if triggered.
            return self.boost_amount
        return 0
