#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'analyse de réseau pour le système Anti-Blanchiment d'Argent (AML).
Fournit des fonctionnalités pour construire et analyser les réseaux de transactions.
"""

import pandas as pd
import numpy as np
import networkx as nx

class NetworkAnalysis:
    """
    Fournit des méthodes pour construire et analyser les réseaux de transactions.
    """
    
    @staticmethod
    def build_transaction_network(transaction_data, entity_data):
        """
        Construire un graphe de réseau de transactions pour l'analyse de réseau.
        
        Paramètres:
        -----------
        transaction_data : DataFrame
            Données de transaction à analyser
        entity_data : DataFrame
            Données d'entité avec informations sur les entités
            
        Retourne:
        --------
        NetworkX DiGraph
            Graphe orienté représentant le réseau de transactions
        """
        if transaction_data is None:
            raise ValueError("Les données de transaction doivent être fournies")
        
        # Créer un graphe orienté
        G = nx.DiGraph()
        
        # Ajouter les nœuds (entités)
        for _, entity in entity_data.iterrows():
            G.add_node(entity['entity_id'], 
                       type=entity['entity_type'],
                       name=entity['entity_name'],
                       country=entity['country'])
        
        # Ajouter les arêtes (transactions)
        for _, txn in transaction_data.iterrows():
            G.add_edge(txn['sender_id'], txn['receiver_id'],
                      amount=txn['amount'],
                      timestamp=txn['timestamp'],
                      transaction_type=txn['transaction_type'])
        
        print(f"Réseau construit avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes")
        
        return G
    
    @staticmethod
    def calculate_network_metrics(network_graph):
        """
        Calculer les métriques de réseau pour chaque entité dans le réseau de transactions.
        
        Paramètres:
        -----------
        network_graph : NetworkX DiGraph
            Graphe du réseau de transactions
            
        Retourne:
        --------
        DataFrame
            DataFrame avec les métriques de réseau pour chaque entité
        """
        if network_graph is None:
            raise ValueError("Le graphe de réseau doit être fourni")
        
        # Calculer la centralité de degré
        in_degree = dict(network_graph.in_degree())
        out_degree = dict(network_graph.out_degree())
        
        # Calculer la centralité d'intermédiarité (identifie les entités qui relient différentes communautés)
        betweenness = nx.betweenness_centrality(network_graph)
        
        # Calculer le PageRank (identifie les entités importantes dans le réseau)
        pagerank = nx.pagerank(network_graph)
        
        # Créer un DataFrame avec les métriques
        network_metrics = pd.DataFrame({
            'entity_id': list(network_graph.nodes()),
            'in_degree': [in_degree.get(n, 0) for n in network_graph.nodes()],
            'out_degree': [out_degree.get(n, 0) for n in network_graph.nodes()],
            'betweenness': [betweenness.get(n, 0) for n in network_graph.nodes()],
            'pagerank': [pagerank.get(n, 0) for n in network_graph.nodes()]
        })
        
        return network_metrics
    
    @staticmethod
    def detect_communities(network_graph):
        """
        Détecter les communautés dans le réseau de transactions.
        
        Paramètres:
        -----------
        network_graph : NetworkX DiGraph
            Graphe du réseau de transactions
            
        Retourne:
        --------
        tuple
            (liste des communautés, dictionnaire de mappage des communautés)
        """
        # Convertir en graphe non orienté pour la détection de communauté
        undirected_graph = network_graph.to_undirected()
        
        # Détecter les communautés en utilisant l'algorithme de Louvain
        communities = nx.community.louvain_communities(undirected_graph)
        
        # Créer un dictionnaire de mappage des communautés
        community_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_mapping[node] = i
        
        return communities, community_mapping