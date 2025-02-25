import networkx as nx
import pandas as pd
import numpy as np
from datetime import timedelta

class TransactionNetworkAnalyzer:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        
    def create_network_features(self, df):
        """Generate network-based features for transaction analysis"""
        df = df.copy()
        
        # Create time-windowed networks
        networks = self._create_transaction_networks(df)
        
        # Calculate node-level metrics
        df = self._add_centrality_features(df, networks)
        
        # Calculate community-based features
        df = self._add_community_features(df, networks)
        
        # Add velocity network features
        df = self._add_velocity_network_features(df)
        
        return df
    
    def _create_transaction_networks(self, df):
        """Create transaction networks for different time windows"""
        networks = {}
        
        for window in [1, 7, self.lookback_days]:
            G = nx.DiGraph()
            
            # Create time window mask
            end_date = df['timestamp'].max()
            start_date = end_date - timedelta(days=window)
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            
            # Build network
            for _, row in df[mask].iterrows():
                sender = f"C_{row['customer_id']}"
                recipient = f"R_{row['recipient_country']}"
                
                # Add edge with transaction properties
                G.add_edge(sender, recipient, 
                          amount=row['amount'],
                          timestamp=row['timestamp'],
                          transaction_type=row['transaction_type'])
                
            networks[window] = G
            
        return networks
    
    def _add_centrality_features(self, df, networks):
        """Add network centrality metrics"""
        for window, G in networks.items():
            # Degree centrality
            in_degree = nx.in_degree_centrality(G)
            out_degree = nx.out_degree_centrality(G)
            
            # Betweenness centrality (identifies bridge nodes)
            betweenness = nx.betweenness_centrality(G)
            
            # PageRank (identifies influential nodes)
            pagerank = nx.pagerank(G)
            
            # Add features to dataframe
            df[f'out_degree_{window}d'] = df['customer_id'].apply(
                lambda x: out_degree.get(f"C_{x}", 0))
            df[f'betweenness_{window}d'] = df['customer_id'].apply(
                lambda x: betweenness.get(f"C_{x}", 0))
            df[f'pagerank_{window}d'] = df['customer_id'].apply(
                lambda x: pagerank.get(f"C_{x}", 0))
            
        return df
    
    def _add_community_features(self, df, networks):
        """Add community detection based features"""
        for window, G in networks.items():
            # Detect communities using Louvain method
            communities = nx.community.louvain_communities(G.to_undirected())
            
            # Create mapping of node to community
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Calculate community-based metrics
            df[f'community_size_{window}d'] = df['customer_id'].apply(
                lambda x: len([c for c in communities if f"C_{x}" in c][0]) 
                if f"C_{x}" in community_map else 0)
            
            # Calculate isolation metrics
            df[f'community_isolation_{window}d'] = df.apply(
                lambda row: self._calculate_isolation_score(
                    G, f"C_{row['customer_id']}", community_map), axis=1)
            
        return df
    
    def _calculate_isolation_score(self, G, node, community_map):
        """Calculate how isolated a node is from its community"""
        if node not in G or node not in community_map:
            return 0
            
        node_community = community_map[node]
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            return 1.0
            
        # Calculate fraction of neighbors in different communities
        different_community = sum(
            1 for n in neighbors 
            if n in community_map and community_map[n] != node_community
        )
        return different_community / len(neighbors)
    
    def _add_velocity_network_features(self, df):
        """Add features based on network velocity patterns"""
        # Calculate network growth rate
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for window in [1, 7, self.lookback_days]:
            df[f'new_connections_{window}d'] = df.groupby('customer_id').apply(
                lambda x: self._calculate_new_connections(x, window)
            ).reset_index(level=0, drop=True)
            
            # Calculate transaction velocity within network
            df[f'network_velocity_{window}d'] = df.groupby('customer_id').apply(
                lambda x: self._calculate_network_velocity(x, window)
            ).reset_index(level=0, drop=True)
            
        return df
    
    def _calculate_new_connections(self, customer_df, window):
        """Calculate rate of new connections for a customer"""
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
        """Calculate transaction velocity within customer's network"""
        end_date = customer_df['timestamp'].max()
        start_date = end_date - timedelta(days=window)
        recent_df = customer_df[customer_df['timestamp'] >= start_date]
        
        if len(recent_df) <= 1:
            return 0
            
        return len(recent_df) / len(set(recent_df['recipient_country']))