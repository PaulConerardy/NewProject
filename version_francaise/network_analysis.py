import pandas as pd
import numpy as np
import networkx as nx

class NetworkAnalysis:
    def __init__(self):
        """Initialise le composant d'analyse de réseau."""
        self.max_score = 15  # Score maximum du composant d'analyse de réseau
    
    def build_transaction_network(self, entities_df, transactions_df, wires_df):
        """Construit un graphe de réseau à partir des données de transactions et de virements."""
        G = nx.DiGraph()
        
        # Ajouter les entités comme nœuds
        for _, entity in entities_df.iterrows():
            G.add_node(entity['entity_id'], 
                       type=entity.get('entity_type', 'unknown'),
                       data=entity)
        
        # Ajouter les transactions comme arêtes
        if transactions_df is not None and not transactions_df.empty:
            for _, txn in transactions_df.iterrows():
                if G.has_node(txn['sender_id']) and G.has_node(txn['destination_id']):
                    G.add_edge(txn['sender_id'], txn['destination_id'], 
                              amount=txn['amount'],
                              date=txn['date'],
                              type=txn['type'],
                              data=txn)
        
        # Ajouter les virements comme arêtes
        if wires_df is not None and not wires_df.empty:
            for _, wire in wires_df.iterrows():
                if G.has_node(wire['sender_id']) and G.has_node(wire['destination_id']):
                    G.add_edge(wire['sender_id'], wire['destination_id'], 
                              amount=wire['amount'],
                              date=wire['date'],
                              type='wire',
                              data=wire)
        
        return G
    
    def detect_circular_flows(self, G, entity_id):
        """Détecte les flux circulaires d'argent impliquant l'entité."""
        score = 0
        
        # Rechercher des cycles qui incluent cette entité
        try:
            cycles = list(nx.simple_cycles(G.subgraph(nx.descendants(G, entity_id) | {entity_id})))
            cycles = [cycle for cycle in cycles if entity_id in cycle]
            
            if len(cycles) > 0:
                score += min(len(cycles), 5)
        except:
            # Gérer le cas où aucun cycle n'existe
            pass
        
        return score
    
    def detect_layering_patterns(self, G, entity_id):
        """Détecte les modèles de stratification (fonds passant par plusieurs entités)."""
        score = 0
        
        # Vérifier les chemins sortants
        try:
            # Trouver des chemins de longueur 2-4 partant de cette entité
            paths = []
            for length in range(2, 5):
                for target in G.nodes():
                    if target != entity_id:
                        for path in nx.all_simple_paths(G, entity_id, target, cutoff=length):
                            if len(path) == length + 1:  # +1 car le chemin inclut les nœuds de début et de fin
                                paths.append(path)
            
            # Score basé sur le nombre de chemins de stratification
            if len(paths) > 10:
                score += 5
            elif len(paths) > 5:
                score += 3
            elif len(paths) > 2:
                score += 1
        except:
            # Gérer les exceptions
            pass
        
        return score
    
    def detect_hub_patterns(self, G, entity_id):
        """Détecte si l'entité agit comme un hub (nombreuses connexions entrantes et sortantes)."""
        score = 0
        
        in_degree = G.in_degree(entity_id)
        out_degree = G.out_degree(entity_id)
        
        # Score basé sur la connectivité
        if in_degree > 5 and out_degree > 5:
            score += 5
        elif in_degree > 3 and out_degree > 3:
            score += 3
        elif in_degree > 1 and out_degree > 1:
            score += 1
        
        return score
    
    def calculate_score(self, entity_id, entities_df, transactions_df, wires_df):
        """Calcule le score d'analyse de réseau pour une entité."""
        # Construire le réseau de transactions
        G = self.build_transaction_network(entities_df, transactions_df, wires_df)
        
        if entity_id not in G:
            return 0
        
        # Appliquer les techniques d'analyse de réseau
        score = 0
        score += self.detect_circular_flows(G, entity_id)
        score += self.detect_layering_patterns(G, entity_id)
        score += self.detect_hub_patterns(G, entity_id)
        
        # Normaliser au max_score
        normalized_score = min(score, self.max_score)
        return normalized_score