import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, output_dir='results/visualizations'):
        """
        Initialise le Visualiseur avec un répertoire de sortie.
        
        Args:
            output_dir (str): Chemin du répertoire où les visualisations seront sauvegardées.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_detection_results(self, alerted_accounts_df, flagged_transactions_df):
        """
        Visualise les résultats de détection du système AML.
        
        Args:
            alerted_accounts_df (pd.DataFrame): DataFrame contenant les informations des comptes alertés.
            flagged_transactions_df (pd.DataFrame): DataFrame contenant les informations des transactions marquées.
        """
        if alerted_accounts_df.empty:
            print("Aucun compte n'a été alerté. Aucune visualisation créée.")
            return
        
        # 1. Créer la distribution des scores
        self._plot_score_distribution(alerted_accounts_df)
        
        # 2. Créer la visualisation des contributions des règles
        self._plot_rule_contributions(alerted_accounts_df)
        
        # 3. Créer la chronologie des transactions marquées si disponible
        if not flagged_transactions_df.empty and 'trx_date' in flagged_transactions_df.columns:
            self._plot_transaction_timeline(flagged_transactions_df)
            
        # 4. Créer la distribution des types de transaction si disponible
        if not flagged_transactions_df.empty and 'transaction_type_desc' in flagged_transactions_df.columns:
            self._plot_transaction_types(flagged_transactions_df)
    
    def _plot_score_distribution(self, alerted_accounts_df):
        """
        Trace la distribution des scores d'alerte.
        
        Args:
            alerted_accounts_df (pd.DataFrame): DataFrame contenant les informations des comptes alertés.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(alerted_accounts_df['total_score'], bins=20, kde=True)
        plt.title('Distribution des Scores d\'Alerte', fontsize=15)
        plt.xlabel('Score d\'Alerte', fontsize=12)
        plt.ylabel('Nombre de Comptes', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'))
        plt.close()
        
        # Créer une version interactive avec Plotly
        fig = px.histogram(
            alerted_accounts_df, 
            x='total_score',
            nbins=20,
            marginal='box',
            title='Distribution des Scores d\'Alerte',
            labels={'total_score': 'Score d\'Alerte'},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(
            xaxis_title='Score d\'Alerte',
            yaxis_title='Nombre de Comptes',
            template='plotly_white'
        )
        fig.write_html(os.path.join(self.output_dir, 'score_distribution.html'))
        
    def _plot_rule_contributions(self, alerted_accounts_df):
        """
        Trace la contribution de chaque règle au total des alertes.
        
        Args:
            alerted_accounts_df (pd.DataFrame): DataFrame contenant les informations des comptes alertés.
        """
        # Extraire les informations des règles de la chaîne JSON
        rule_counts = {}
        for triggered_rules in alerted_accounts_df['triggered_rules']:
            rules_dict = json.loads(triggered_rules)
            for rule, score in rules_dict.items():
                if rule not in rule_counts:
                    rule_counts[rule] = 0
                if score > 0:
                    rule_counts[rule] += 1
        
        # Convertir en DataFrame pour la visualisation
        rule_df = pd.DataFrame({
            'Règle': list(rule_counts.keys()),
            'Nombre': list(rule_counts.values())
        }).sort_values('Nombre', ascending=False)
        
        # Créer la visualisation matplotlib
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Nombre', y='Règle', data=rule_df, palette='viridis')
        plt.title('Nombre d\'Alertes par Règle', fontsize=15)
        plt.xlabel('Nombre de Comptes Déclenchés', fontsize=12)
        plt.ylabel('Règle', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rule_contributions.png'))
        plt.close()
        
        # Créer une version interactive avec Plotly
        fig = px.bar(
            rule_df, 
            x='Nombre', 
            y='Règle',
            title='Nombre d\'Alertes par Règle',
            orientation='h',
            color='Nombre',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Nombre de Comptes Déclenchés',
            yaxis_title='Règle',
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.write_html(os.path.join(self.output_dir, 'rule_contributions.html'))
    
    def _plot_transaction_timeline(self, flagged_transactions_df):
        """
        Trace la chronologie des transactions marquées.
        
        Args:
            flagged_transactions_df (pd.DataFrame): DataFrame contenant les informations des transactions marquées.
        """
        # S'assurer que la colonne de date est en datetime
        transactions = flagged_transactions_df.copy()
        try:
            transactions['trx_date'] = pd.to_datetime(transactions['trx_date'])
        except:
            # Si la conversion échoue, ignorer cette visualisation
            print("Impossible de convertir les dates des transactions. Visualisation de la chronologie ignorée.")
            return
        
        # Grouper par date et compter
        daily_counts = transactions.groupby(transactions['trx_date'].dt.date).size().reset_index(name='count')
        
        # Créer la visualisation matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(daily_counts['trx_date'], daily_counts['count'], marker='o', linestyle='-')
        plt.title('Transactions Marquées dans le Temps', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Nombre de Transactions Marquées', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'transaction_timeline.png'))
        plt.close()
        
        # Créer une version interactive avec Plotly
        fig = px.line(
            daily_counts, 
            x='trx_date', 
            y='count',
            title='Transactions Marquées dans le Temps',
            labels={'trx_date': 'Date', 'count': 'Nombre de Transactions Marquées'},
            markers=True
        )
        fig.update_layout(template='plotly_white')
        fig.write_html(os.path.join(self.output_dir, 'transaction_timeline.html'))
    
    def _plot_transaction_types(self, flagged_transactions_df):
        """
        Trace la distribution des types de transaction parmi les transactions marquées.
        
        Args:
            flagged_transactions_df (pd.DataFrame): DataFrame contenant les informations des transactions marquées.
        """
        if 'transaction_type_desc' not in flagged_transactions_df.columns:
            return
            
        # Grouper par type de transaction et compter
        type_counts = flagged_transactions_df['transaction_type_desc'].value_counts().reset_index()
        type_counts.columns = ['Type de Transaction', 'Nombre']
        
        # Créer la visualisation matplotlib
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Nombre', y='Type de Transaction', data=type_counts, palette='viridis')
        plt.title('Transactions Marquées par Type', fontsize=15)
        plt.xlabel('Nombre de Transactions', fontsize=12)
        plt.ylabel('Type de Transaction', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'transaction_types.png'))
        plt.close()
        
        # Créer une version interactive avec Plotly
        fig = px.bar(
            type_counts, 
            x='Nombre', 
            y='Type de Transaction',
            title='Transactions Marquées par Type',
            orientation='h',
            color='Nombre',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Nombre de Transactions',
            yaxis_title='Type de Transaction',
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.write_html(os.path.join(self.output_dir, 'transaction_types.html'))

    def visualize_simulation_results(self, results_csv_path='simulation_results/param_optimization_results.csv'):
        """
        Visualise les résultats de simulation de l'optimisation des paramètres.
        
        Args:
            results_csv_path (str): Chemin vers le fichier CSV des résultats de simulation.
        """
        # Charger les résultats de simulation
        if not os.path.exists(results_csv_path):
            print(f"Fichier de résultats {results_csv_path} non trouvé.")
            return
            
        results_df = pd.read_csv(results_csv_path)
        
        # 1. Visualisation des métriques de performance
        self._plot_performance_metrics(results_df)
        
        # 2. Visualisation de l'impact des paramètres
        self._plot_parameter_impact(results_df)
    
    def _plot_performance_metrics(self, results_df):
        """
        Trace les métriques de performance des résultats de simulation.
        
        Args:
            results_df (pd.DataFrame): DataFrame contenant les résultats de simulation.
        """
        # Créer un graphique de coordonnées parallèles
        if all(col in results_df.columns for col in ['precision', 'recall', 'f1', 'num_alerts']):
            # Créer une version interactive avec Plotly
            fig = px.parallel_coordinates(
                results_df,
                dimensions=['precision', 'recall', 'f1', 'num_alerts'],
                color='f1',
                color_continuous_scale='Viridis',
                title='Métriques de Performance à travers les Simulations'
            )
            fig.update_layout(template='plotly_white')
            fig.write_html(os.path.join(self.output_dir, 'performance_metrics.html'))
        
            # Créer un nuage de points précision vs rappel
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                results_df['precision'], 
                results_df['recall'], 
                c=results_df['f1'], 
                s=results_df['num_alerts']/5,
                alpha=0.7,
                cmap='viridis'
            )
            plt.colorbar(scatter, label='Score F1')
            plt.title('Précision vs Rappel à travers les Simulations', fontsize=15)
            plt.xlabel('Précision', fontsize=12)
            plt.ylabel('Rappel', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'precision_recall.png'))
            plt.close()
            
            # Version interactive
            fig = px.scatter(
                results_df,
                x='precision',
                y='recall',
                color='f1',
                size='num_alerts',
                hover_data=['prior_suspicious_flag_boost'],
                title='Précision vs Rappel à travers les Simulations'
            )
            fig.update_layout(template='plotly_white')
            fig.write_html(os.path.join(self.output_dir, 'precision_recall.html'))
    
    def _plot_parameter_impact(self, results_df):
        """
        Trace l'impact des différents paramètres sur la performance.
        
        Args:
            results_df (pd.DataFrame): DataFrame contenant les résultats de simulation.
        """
        # Tracer l'impact du bonus de drapeau suspect antérieur
        if 'prior_suspicious_flag_boost' in results_df.columns and 'f1' in results_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='prior_suspicious_flag_boost', y='f1', data=results_df)
            plt.title('Impact du Bonus de Drapeau Suspect sur le Score F1', fontsize=15)
            plt.xlabel('Bonus de Drapeau Suspect', fontsize=12)
            plt.ylabel('Score F1', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'prior_flag_impact.png'))
            plt.close()
            
            # Version interactive
            fig = px.box(
                results_df,
                x='prior_suspicious_flag_boost',
                y='f1',
                title='Impact du Bonus de Drapeau Suspect sur le Score F1'
            )
            fig.update_layout(template='plotly_white')
            fig.write_html(os.path.join(self.output_dir, 'prior_flag_impact.html'))
            
        # Créer un tableau de bord interactif récapitulatif
        if all(col in results_df.columns for col in ['precision', 'recall', 'f1', 'num_alerts']):
            # Trouver le meilleur jeu de paramètres
            best_row = results_df.loc[results_df['f1'].idxmax()]
            
            # Créer un tableau de bord avec plusieurs graphiques
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Distribution du Score F1', 
                    'Précision vs Rappel',
                    'Distribution du Nombre d\'Alertes',
                    'Impact du Bonus de Drapeau Suspect'
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "box"}]
                ]
            )
            
            # Distribution du Score F1
            fig.add_trace(
                go.Histogram(x=results_df['f1'], name='Score F1'),
                row=1, col=1
            )
            
            # Précision vs Rappel
            fig.add_trace(
                go.Scatter(
                    x=results_df['precision'], 
                    y=results_df['recall'], 
                    mode='markers',
                    marker=dict(
                        size=results_df['num_alerts']/10,
                        color=results_df['f1'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Score F1')
                    ),
                    name='Simulations'
                ),
                row=1, col=2
            )
            
            # Distribution du Nombre d'Alertes
            fig.add_trace(
                go.Histogram(x=results_df['num_alerts'], name='Nombre d\'Alertes'),
                row=2, col=1
            )
            
            # Impact du Bonus de Drapeau Suspect
            if 'prior_suspicious_flag_boost' in results_df.columns:
                for boost_value in results_df['prior_suspicious_flag_boost'].unique():
                    subset = results_df[results_df['prior_suspicious_flag_boost'] == boost_value]
                    fig.add_trace(
                        go.Box(
                            y=subset['f1'], 
                            name=str(boost_value),
                            boxmean=True
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title='Tableau de Bord des Résultats de Simulation',
                template='plotly_white',
                height=900,
                width=1200,
                showlegend=False
            )
            
            fig.write_html(os.path.join(self.output_dir, 'simulation_dashboard.html'))


# Exemple d'utilisation dans detection_runner.py:
# from visualization import Visualizer
# visualizer = Visualizer()
# visualizer.visualize_detection_results(alerted_accounts_df, flagged_transactions_df)

# Exemple d'utilisation dans simulator.py:
# from visualization import Visualizer
# visualizer = Visualizer()
# visualizer.visualize_simulation_results() 