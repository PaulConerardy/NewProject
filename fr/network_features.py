import networkx as nx
import pandas as pd
import numpy as np
from datetime import timedelta

class TransactionNetworkAnalyzer:
    """
    Analyseur de réseau de transactions financières.
    
    Cette classe permet d'analyser les relations entre les entités dans un réseau
    de transactions et génère des caractéristiques basées sur la théorie des graphes.
    
    Attributs:
        lookback_days (int): Nombre de jours d'historique à considérer
    """
    
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        
    def create_network_features(self, df):
        """
        Génère les caractéristiques basées sur l'analyse de réseau.
        
        Crée un ensemble complet de métriques de réseau incluant:
        - Centralité des nœuds
        - Détection de communautés
        - Métriques de vélocité du réseau
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les caractéristiques de réseau
        """
        df = df.copy()
        
        # Création des réseaux temporels
        networks = self._create_transaction_networks(df)
        
        # Calcul des métriques au niveau des nœuds
        df = self._add_centrality_features(df, networks)
        
        # Calcul des caractéristiques basées sur les communautés
        df = self._add_community_features(df, networks)
        
        # Ajout des caractéristiques de vélocité du réseau
        df = self._add_velocity_network_features(df)
        
        return df
    
    def _create_transaction_networks(self, df):
        """
        Crée des réseaux de transactions pour différentes fenêtres temporelles.
        
        Construit des graphes dirigés où:
        - Les nœuds représentent les clients et les destinataires
        - Les arêtes représentent les transactions
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            dict: Dictionnaire des réseaux par fenêtre temporelle
        """
        networks = {}
        
        for window in [1, 7, self.lookback_days]:
            G = nx.DiGraph()
            
            # Création du masque temporel
            end_date = df['timestamp'].max()
            start_date = end_date - timedelta(days=window)
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            
            # Construction du réseau
            for _, row in df[mask].iterrows():
                sender = f"C_{row['ref_id']}"
                recipient = f"R_{row['recipient_country']}"
                
                # Ajout de l'arête avec les propriétés de la transaction
                G.add_edge(sender, recipient, 
                          amount=row['amount'],
                          timestamp=row['timestamp'],
                          transaction_type=row['transaction_type'])
                
            networks[window] = G
            
        return networks
    
    def _add_centrality_features(self, df, networks):
        """
        Ajoute les métriques de centralité du réseau.
        
        Calcule différentes mesures de centralité:
        - Centralité de degré
        - Centralité d'intermédiarité
        - PageRank
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            networks (dict): Réseaux de transactions
            
        Returns:
            pd.DataFrame: DataFrame avec métriques de centralité
        """
        for window, G in networks.items():
            # Centralité de degré
            in_degree = nx.in_degree_centrality(G)
            out_degree = nx.out_degree_centrality(G)
            
            # Centralité d'intermédiarité (identifie les nœuds ponts)
            betweenness = nx.betweenness_centrality(G)
            
            # PageRank (identifie les nœuds influents)
            pagerank = nx.pagerank(G)
            
            # Ajout des caractéristiques au DataFrame
            df[f'out_degree_{window}d'] = df['ref_id'].apply(
                lambda x: out_degree.get(f"C_{x}", 0))
            df[f'betweenness_{window}d'] = df['ref_id'].apply(
                lambda x: betweenness.get(f"C_{x}", 0))
            df[f'pagerank_{window}d'] = df['ref_id'].apply(
                lambda x: pagerank.get(f"C_{x}", 0))
            
        return df
    
    def _add_community_features(self, df, networks):
        """
        Ajoute les caractéristiques basées sur la détection de communautés.
        
        Analyse:
        - Taille des communautés
        - Isolation des nœuds
        - Structure des communautés
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            networks (dict): Réseaux de transactions
            
        Returns:
            pd.DataFrame: DataFrame avec métriques de communauté
        """
        for window, G in networks.items():
            # Détection des communautés avec la méthode Louvain
            communities = nx.community.louvain_communities(G.to_undirected())
            
            # Création de la cartographie nœud-communauté
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Calcul des métriques basées sur les communautés
            df[f'community_size_{window}d'] = df['ref_id'].apply(
                lambda x: len([c for c in communities if f"C_{x}" in c][0]) 
                if f"C_{x}" in community_map else 0)
            
            # Calcul des métriques d'isolation
            df[f'community_isolation_{window}d'] = df.apply(
                lambda row: self._calculate_isolation_score(
                    G, f"C_{row['ref_id']}", community_map), axis=1)
            
        return df
    
    def _calculate_isolation_score(self, G, node, community_map):
        """
        Calcule le degré d'isolation d'un nœud par rapport à sa communauté.
        
        Mesure:
        - La proportion de voisins dans d'autres communautés
        - L'isolation relative du nœud
        
        Args:
            G (nx.Graph): Graphe de transactions
            node (str): Identifiant du nœud
            community_map (dict): Cartographie des communautés
            
        Returns:
            float: Score d'isolation (0-1)
        """
        if node not in G or node not in community_map:
            return 0
            
        node_community = community_map[node]
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            return 1.0
            
        # Calcul de la fraction de voisins dans différentes communautés
        different_community = sum(
            1 for n in neighbors 
            if n in community_map and community_map[n] != node_community
        )
        return different_community / len(neighbors)
    
    def _add_velocity_network_features(self, df):
        """
        Ajoute les caractéristiques basées sur la vélocité du réseau.
        
        Analyse:
        - Taux de croissance du réseau
        - Nouvelles connexions
        - Vélocité des transactions
        
        Args:
            df (pd.DataFrame): DataFrame des transactions
            
        Returns:
            pd.DataFrame: DataFrame avec métriques de vélocité
        """
        # Calcul du taux de croissance du réseau
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for window in [1, 7, self.lookback_days]:
            df[f'new_connections_{window}d'] = df.groupby('ref_id').apply(
                lambda x: self._calculate_new_connections(x, window)
            ).reset_index(level=0, drop=True)
            
            # Calcul de la vélocité des transactions dans le réseau
            df[f'network_velocity_{window}d'] = df.groupby('ref_id').apply(
                lambda x: self._calculate_network_velocity(x, window)
            ).reset_index(level=0, drop=True)
            
        return df
    
    def _calculate_new_connections(self, customer_df, window):
        """
        Calcule le taux de nouvelles connexions pour un client.
        
        Mesure:
        - Nombre de nouveaux destinataires
        - Expansion du réseau
        
        Args:
            customer_df (pd.DataFrame): DataFrame des transactions d'un client
            window (int): Fenêtre temporelle en jours
            
        Returns:
            int: Nombre de nouvelles connexions
        """
        end_date = customer_df['timestamp'].max()
        start_date = end_date - timedelta(days=window)
        
        recent_recipients = set(customer_df[
            customer_df['timestamp'] >= start_date
        ]['recipient_country'])
        
        previous_recipients = set(customer_df[
            customer_df['timestamp'] < start_date
        ]['recipient_country'])
        
        return len(recent_recipients - previous_recipients)
    
    def _calculate_network_velocity(self, customer_df, window):
        """
        Calcule la vélocité des transactions dans le réseau d'un client.
        
        Mesure:
        - Fréquence des transactions
        - Diversité des destinataires
        
        Args:
            customer_df (pd.DataFrame): DataFrame des transactions d'un client
            window (int): Fenêtre temporelle en jours
            
        Returns:
            float: Score de vélocité du réseau
        """
        end_date = customer_df['timestamp'].max()
        start_date = end_date - timedelta(days=window)
        recent_df = customer_df[customer_df['timestamp'] >= start_date]
        
        if len(recent_df) <= 1:
            return 0
            
        return len(recent_df) / len(set(recent_df['recipient_country']))