import pandas as pd
import os
import json
from visualization import Visualizer

class Alert:
    def __init__(self, output_paths, entity_df, visualizer):
        """
        Initialise l'enregistreur d'alertes avec les chemins de sortie, les données d'entités et le visualiseur.

        Args:
            output_paths (dict): Dictionnaire contenant les chemins vers les fichiers de sortie.
            entity_df (DataFrame): Données des entités pour obtenir les noms et autres infos client.
            visualizer (Visualizer): Instance du visualiseur.
        """
        self.output_paths = output_paths
        self.entity_df = entity_df
        self.visualizer = visualizer
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Créer les répertoires de sortie s'ils n'existent pas."""
        for path in self.output_paths.values():
            output_dir = os.path.dirname(path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

    def save_alerts(self, alerted_accounts_df, flagged_transactions_df):
        """
        Sauvegarder les résultats de détection dans les fichiers de sortie au format CSV délimité par '|'.
        Le fichier des comptes alertés est formaté par client.

        Args:
            alerted_accounts_df (DataFrame): Données des comptes alertés.
            flagged_transactions_df (DataFrame): Données des transactions marquées.
        """
        if not alerted_accounts_df.empty:
            # Fusionner avec entity_df pour obtenir le nom du client
            # Ajuster les noms de colonnes ('party_key', 'name') en fonction de la structure réelle d'entity_df si nécessaire
            alerted_accounts_with_names = alerted_accounts_df.merge(
                self.entity_df[['party_key', 'name']], on='party_key', how='left'
            )

            # Grouper par client (party_key) et agréger les informations
            client_alerts = alerted_accounts_with_names.groupby('party_key').agg(
                client_name=('name', 'first'),
                accounts=('account_key', lambda x: ', '.join(x.astype(str))),
                client_risk=('total_score', 'max'), # Utiliser le score max comme risque client représentatif
                triggered_rules_summary=('triggered_rules', lambda x: ' | '.join([', '.join(json.loads(rules_str).keys()) for rules_str in x]))
            ).reset_index()

            # Renommer les colonnes en français
            client_alerts = client_alerts.rename(columns={
                'party_key': 'Client ID',
                'client_name': 'Nom Client',
                'accounts': 'Comptes',
                'client_risk': 'Risque Client',
                'triggered_rules_summary': 'Résumé Motifs Suspicieux'
            })

            # Sauvegarder les alertes agrégées par client au format CSV
            client_alerts.to_csv(self.output_paths['alerted_accounts'], index=False, sep='|')
            print(f"Comptes alertés (par client) sauvegardés dans {self.output_paths['alerted_accounts']}")
        else:
            print("Aucun compte n'a été alerté.")

        # Sauvegarder les transactions marquées au format spécifié
        if not flagged_transactions_df.empty:
            # Sélectionner et renommer les colonnes en fonction du format demandé.
            # Supposons que flagged_transactions_df contient des colonnes comme :
            # 'transaction_key', 'amount', 'transaction_code', 'party_key', 'account_key',
            # 'beneficiary', 'sender', 'narrative'.
            # Ajuster les noms de colonnes ici si votre DataFrame réel a des noms différents.
            output_cols = [
                'transaction_key',
                'amount',
                'transaction_code',
                'party_key',
                'account_key',
                'beneficiary',
                'sender',
                'narrative'
                ]

            # Filtrer le DataFrame pour inclure uniquement les colonnes souhaitées dans la sortie, si elles existent
            # Cela évite les erreurs si certaines colonnes (comme beneficiary/sender pour les non-virements) sont manquantes
            cols_to_save = [col for col in output_cols if col in flagged_transactions_df.columns]
            flagged_transactions_output_df = flagged_transactions_df[cols_to_save].copy()

            # Renommer les colonnes en français
            column_rename_map = {
                'transaction_key': 'Clé Transaction',
                'amount': 'Montant',
                'transaction_code': 'Code Transaction',
                'party_key': 'ID Client Lié',
                'account_key': 'Compte Lié',
                'beneficiary': 'Nom Bénéficiaire',
                'sender': 'Nom Expéditeur',
                'narrative': 'Narratif'
            }

            # Appliquer le renommage uniquement aux colonnes qui existent et sont sélectionnées pour la sortie
            columns_to_rename = {k: v for k, v in column_rename_map.items() if k in flagged_transactions_output_df.columns}
            flagged_transactions_output_df = flagged_transactions_output_df.rename(columns=columns_to_rename)

            flagged_transactions_output_df.to_csv(self.output_paths['flagged_transactions'], index=False, sep='|')
            print(f"Transactions marquées sauvegardées (par transaction) dans {self.output_paths['flagged_transactions']}")
        else:
            print("Aucune transaction n'a été marquée.")

    def generate_visualizations(self, alerted_accounts_df, flagged_transactions_df):
        """
        Générer les visualisations des résultats de détection.

        Args:
            alerted_accounts_df (DataFrame): Données des comptes alertés.
            flagged_transactions_df (DataFrame): Données des transactions marquées.
        """
        if not alerted_accounts_df.empty:
            print("Génération des visualisations...")
            self.visualizer.visualize_detection_results(alerted_accounts_df, flagged_transactions_df)
            print(f"Visualisations sauvegardées dans {self.visualizer.output_dir}")
        else:
            print("Aucune visualisation générée car aucun compte n'a été alerté.")
        