import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import time

class NetworkAnalysisDetection:
    """
    Module de détection basé sur l'analyse de réseaux pour identifier les schémas de blanchiment d'argent,
    en particulier les réseaux de mules financières et les systèmes bancaires clandestins.
    """
    
    def __init__(self):
        """Initialise le composant de détection basée sur l'analyse de réseaux."""
        self.max_score = 30  # Score maximum du composant
        self.threshold_amount = 10000  # Seuil de déclaration standard
        self.suspicious_countries = ['IR', 'AE', 'KW', 'HK', 'CN']  # Codes pays à risque
        
        # Paramètres pour l'optimisation des performances
        self.min_transaction_amount = 1000  # Montant minimum pour considérer une transaction dans l'analyse de réseau
        self.max_network_size = 1000  # Taille maximale du réseau à analyser pour éviter les problèmes de performance
        self.time_window_days = 90  # Fenêtre temporelle pour l'analyse (en jours)
        
        # Poids des différentes métriques de réseau
        self.network_metrics = {
            "betweenness_centrality": 10,  # Entités qui servent d'intermédiaires dans le réseau
            "degree_centrality": 8,        # Entités avec beaucoup de connexions
            "community_outlier": 7,        # Entités qui se comportent différemment de leur communauté
            "transaction_velocity": 10,    # Vitesse des transactions (entrée-sortie rapide)
            "structured_patterns": 8       # Schémas de transactions structurées
        }
    
    def preprocess_data(self, transactions_df, wires_df, entity_df):
        """
        Prétraite les données pour l'analyse de réseau en filtrant et en formatant les transactions.
        Optimise la performance en réduisant le volume de données à analyser.
        """
        start_time = time.time()
        
        # Convertir les dates en datetime
        if transactions_df is not None and not transactions_df.empty:
            transactions_df['trx_date'] = pd.to_datetime(transactions_df['trx_date'], format='%d%b%Y', errors='coerce')
            
            # Filtrer les transactions récentes pour limiter le volume de données
            if 'trx_date' in transactions_df.columns:
                latest_date = transactions_df['trx_date'].max()
                cutoff_date = latest_date - pd.Timedelta(days=self.time_window_days)
                transactions_df = transactions_df[transactions_df['trx_date'] >= cutoff_date]
            
            # Filtrer les transactions de faible montant pour réduire le bruit
            transactions_df = transactions_df[transactions_df['amount'] >= self.min_transaction_amount]
        
        if wires_df is not None and not wires_df.empty:
            wires_df['wire_date'] = pd.to_datetime(wires_df['wire_date'], format='%d%b%Y', errors='coerce')
            
            # Filtrer les virements récents
            if 'wire_date' in wires_df.columns:
                latest_date = wires_df['wire_date'].max()
                cutoff_date = latest_date - pd.Timedelta(days=self.time_window_days)
                wires_df = wires_df[wires_df['wire_date'] >= cutoff_date]
            
            # Filtrer les virements de faible montant
            wires_df = wires_df[wires_df['amount'] >= self.min_transaction_amount]
        
        print(f"Prétraitement des données terminé en {time.time() - start_time:.2f} secondes")
        return transactions_df, wires_df
    
    def build_transaction_network(self, transactions_df, wires_df):
        """
        Construit un réseau de transactions à partir des données de transactions et de virements.
        Optimisé pour la performance avec de grands volumes de données.
        """
        start_time = time.time()
        
        # Initialiser le graphe
        G = nx.DiGraph()
        
        # Ajouter les transactions comme arêtes dans le graphe
        if transactions_df is not None and not transactions_df.empty:
            # Pour les transactions, nous devons inférer la source et la destination
            # car les données ne contiennent pas explicitement cette information
            
            # Regrouper les transactions par date et par entité pour détecter les flux
            transactions_df = transactions_df.sort_values('trx_date')
            
            # Créer un dictionnaire pour stocker les transactions par jour et par entité
            daily_transactions = defaultdict(list)
            
            # Parcourir les transactions et les regrouper par jour
            for _, row in transactions_df.iterrows():
                entity_id = row['party_key']
                date = row['trx_date'].date() if pd.notna(row['trx_date']) else None
                if date:
                    daily_transactions[(date, entity_id)].append(row)
            
            # Analyser les flux d'argent pour chaque jour
            for (date, entity_id), txns in daily_transactions.items():
                # Séparer les entrées et sorties
                inflows = [t for t in txns if t['sign'] == '+']
                outflows = [t for t in txns if t['sign'] == '-']
                
                # Si une entité a des entrées et des sorties le même jour, créer des liens potentiels
                if inflows and outflows:
                    # Ajouter l'entité au graphe si elle n'existe pas déjà
                    if not G.has_node(entity_id):
                        G.add_node(entity_id, type='entity')
                    
                    # Pour chaque entrée, créer un lien "source inconnue" -> entité
                    for inflow in inflows:
                        source = f"unknown_source_{inflow['transaction_id']}"
                        G.add_node(source, type='unknown')
                        G.add_edge(source, entity_id, 
                                  amount=inflow['amount'],
                                  date=date,
                                  transaction_type=inflow['transaction_type_desc'])
                    
                    # Pour chaque sortie, créer un lien entité -> "destination inconnue"
                    for outflow in outflows:
                        target = f"unknown_target_{outflow['transaction_id']}"
                        G.add_node(target, type='unknown')
                        G.add_edge(entity_id, target, 
                                  amount=outflow['amount'],
                                  date=date,
                                  transaction_type=outflow['transaction_type_desc'])
        
        # Ajouter les virements comme arêtes dans le graphe
        if wires_df is not None and not wires_df.empty:
            for _, row in wires_df.iterrows():
                source = row['originator_key'] if 'originator_key' in row.index else row['party_key']
                target = row['beneficiary_key'] if 'beneficiary_key' in row.index else None
                
                # Si nous n'avons pas d'information sur le bénéficiaire, utiliser un nœud générique
                if target is None or pd.isna(target):
                    if row['sign'] == '+':
                        # C'est un virement entrant, la source est connue mais pas dans nos données
                        source = f"external_{row['originator']}"
                        target = row['party_key']
                    else:
                        # C'est un virement sortant, la destination est connue mais pas dans nos données
                        source = row['party_key']
                        target = f"external_{row['beneficiary']}"
                
                # Ajouter les nœuds s'ils n'existent pas déjà
                if not G.has_node(source):
                    G.add_node(source, type='entity' if isinstance(source, int) else 'external')
                if not G.has_node(target):
                    G.add_node(target, type='entity' if isinstance(target, int) else 'external')
                
                # Ajouter l'arête avec les attributs du virement
                G.add_edge(source, target, 
                          amount=row['amount'],
                          date=row['wire_date'].date() if pd.notna(row['wire_date']) else None,
                          transaction_type='wire',
                          country_from=row['originator_country'] if 'originator_country' in row.index else None,
                          country_to=row['beneficiary_country'] if 'beneficiary_country' in row.index else None)
        
        # Limiter la taille du réseau pour des raisons de performance
        if len(G) > self.max_network_size:
            print(f"Réseau trop grand ({len(G)} nœuds), réduction à {self.max_network_size} nœuds")
            # Garder les nœuds avec le plus de connexions
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:self.max_network_size]
            G = G.subgraph(top_nodes).copy()
        
        print(f"Construction du réseau terminée en {time.time() - start_time:.2f} secondes")
        print(f"Réseau construit avec {len(G.nodes())} nœuds et {len(G.edges())} arêtes")
        
        return G
    
    def calculate_network_metrics(self, G, entity_id):
        """
        Calcule les métriques de réseau pour une entité spécifique.
        Optimisé pour la performance en limitant les calculs aux métriques essentielles.
        """
        start_time = time.time()
        
        # Vérifier si l'entité existe dans le réseau
        if not G.has_node(entity_id):
            return {}
        
        metrics = {}
        
        # Extraire le sous-graphe pertinent pour l'entité (voisinage à 2 sauts)
        # Cela améliore considérablement les performances pour les grands réseaux
        neighbors = list(nx.neighbors(G, entity_id))
        neighbors.extend([n for nbr in neighbors for n in nx.neighbors(G, nbr)])
        neighbors = list(set(neighbors))  # Éliminer les doublons
        neighbors.append(entity_id)
        subgraph = G.subgraph(neighbors).copy()
        
        # 1. Centralité d'intermédiarité (betweenness centrality)
        # Mesure à quel point un nœud sert d'intermédiaire dans le réseau
        try:
            # Utiliser une approximation pour les grands réseaux
            if len(subgraph) > 100:
                bc = nx.betweenness_centrality(subgraph, k=min(50, len(subgraph) - 1), normalized=True)
            else:
                bc = nx.betweenness_centrality(subgraph, normalized=True)
            metrics['betweenness_centrality'] = bc.get(entity_id, 0)
        except:
            metrics['betweenness_centrality'] = 0
        
        # 2. Centralité de degré (degree centrality)
        # Mesure le nombre de connexions directes
        try:
            in_degree = G.in_degree(entity_id)
            out_degree = G.out_degree(entity_id)
            metrics['in_degree'] = in_degree
            metrics['out_degree'] = out_degree
            metrics['degree_centrality'] = (in_degree + out_degree) / (len(G) - 1) if len(G) > 1 else 0
        except:
            metrics['in_degree'] = 0
            metrics['out_degree'] = 0
            metrics['degree_centrality'] = 0
        
        # 3. Détection de communauté et analyse des valeurs aberrantes
        # Identifier si l'entité se comporte différemment de sa communauté
        try:
            if len(subgraph) > 10:
                communities = nx.community.greedy_modularity_communities(subgraph.to_undirected())
                entity_community = None
                for i, comm in enumerate(communities):
                    if entity_id in comm:
                        entity_community = i
                        break
                
                if entity_community is not None:
                    # Calculer les statistiques de la communauté
                    community_nodes = list(communities[entity_community])
                    community_in_degrees = [G.in_degree(n) for n in community_nodes if n != entity_id]
                    community_out_degrees = [G.out_degree(n) for n in community_nodes if n != entity_id]
                    
                    # Calculer les moyennes et écarts-types
                    if community_in_degrees:
                        avg_in = sum(community_in_degrees) / len(community_in_degrees)
                        std_in = np.std(community_in_degrees) if len(community_in_degrees) > 1 else 0
                        if std_in > 0:
                            metrics['in_degree_zscore'] = (metrics['in_degree'] - avg_in) / std_in
                        else:
                            metrics['in_degree_zscore'] = 0
                    else:
                        metrics['in_degree_zscore'] = 0
                    
                    if community_out_degrees:
                        avg_out = sum(community_out_degrees) / len(community_out_degrees)
                        std_out = np.std(community_out_degrees) if len(community_out_degrees) > 1 else 0
                        if std_out > 0:
                            metrics['out_degree_zscore'] = (metrics['out_degree'] - avg_out) / std_out
                        else:
                            metrics['out_degree_zscore'] = 0
                    else:
                        metrics['out_degree_zscore'] = 0
                    
                    # Déterminer si l'entité est une valeur aberrante dans sa communauté
                    metrics['community_outlier'] = max(abs(metrics.get('in_degree_zscore', 0)), 
                                                     abs(metrics.get('out_degree_zscore', 0)))
                else:
                    metrics['community_outlier'] = 0
            else:
                metrics['community_outlier'] = 0
        except:
            metrics['community_outlier'] = 0
        
        # 4. Vitesse de transaction (transaction velocity)
        # Mesure la rapidité avec laquelle l'argent entre et sort
        try:
            in_edges = list(G.in_edges(entity_id, data=True))
            out_edges = list(G.out_edges(entity_id, data=True))
            
            # Calculer les dates des transactions entrantes et sortantes
            in_dates = [e[2]['date'] for e in in_edges if 'date' in e[2] and e[2]['date'] is not None]
            out_dates = [e[2]['date'] for e in out_edges if 'date' in e[2] and e[2]['date'] is not None]
            
            if in_dates and out_dates:
                # Trier les dates
                in_dates.sort()
                out_dates.sort()
                
                # Calculer le délai moyen entre entrée et sortie
                velocity_scores = []
                for in_date in in_dates:
                    # Trouver la première date de sortie après cette entrée
                    future_out_dates = [d for d in out_dates if d >= in_date]
                    if future_out_dates:
                        # Calculer le délai en jours
                        delay = (min(future_out_dates) - in_date).days
                        # Plus le délai est court, plus le score est élevé
                        velocity_score = max(0, 10 - delay) / 10 if delay <= 10 else 0
                        velocity_scores.append(velocity_score)
                
                # Moyenne des scores de vélocité
                metrics['transaction_velocity'] = sum(velocity_scores) / len(velocity_scores) if velocity_scores else 0
            else:
                metrics['transaction_velocity'] = 0
        except:
            metrics['transaction_velocity'] = 0
        
        # 5. Schémas de transactions structurées
        # Détecter les schémas de transactions juste en dessous du seuil de déclaration
        try:
            # Extraire les montants des transactions
            in_amounts = [e[2]['amount'] for e in in_edges if 'amount' in e[2]]
            out_amounts = [e[2]['amount'] for e in out_edges if 'amount' in e[2]]
            
            # Compter les transactions juste en dessous du seuil
            threshold = self.threshold_amount
            margin = 1000  # Marge en dessous du seuil
            
            structured_in = sum(1 for a in in_amounts if threshold - margin <= a < threshold)
            structured_out = sum(1 for a in out_amounts if threshold - margin <= a < threshold)
            
            # Normaliser par le nombre total de transactions
            total_in = len(in_amounts)
            total_out = len(out_amounts)
            
            if total_in > 0:
                structured_in_ratio = structured_in / total_in
            else:
                structured_in_ratio = 0
                
            if total_out > 0:
                structured_out_ratio = structured_out / total_out
            else:
                structured_out_ratio = 0
            
            # Combiner les ratios
            metrics['structured_patterns'] = max(structured_in_ratio, structured_out_ratio)
        except:
            metrics['structured_patterns'] = 0
        
        print(f"Calcul des métriques pour l'entité {entity_id} terminé en {time.time() - start_time:.2f} secondes")
        return metrics
    
    def detect_suspicious_patterns(self, G, entity_id, entity_df):
        """
        Détecte les schémas suspects spécifiques aux ESM et banques clandestines
        en utilisant l'analyse de réseau.
        """
        patterns = {}
        
        # Vérifier si l'entité existe dans le réseau
        if not G.has_node(entity_id):
            return patterns
        
        # 1. Détection de mules d'argent
        # Caractérisé par des entités qui reçoivent des fonds de multiples sources
        # et les transfèrent rapidement à d'autres entités
        in_edges = list(G.in_edges(entity_id, data=True))
        out_edges = list(G.out_edges(entity_id, data=True))
        
        unique_in_sources = len(set(e[0] for e in in_edges))
        unique_out_targets = len(set(e[1] for e in out_edges))
        
        # Une mule typique a plusieurs sources et plusieurs destinations
        if unique_in_sources >= 3 and unique_out_targets >= 2:
            patterns['potential_money_mule'] = True
        else:
            patterns['potential_money_mule'] = False
        
        # 2. Détection de réseaux de structuration
        # Caractérisé par plusieurs entités qui font des transactions structurées
        # et qui sont connectées entre elles
        neighbors = list(nx.all_neighbors(G, entity_id))
        
        # Vérifier si les voisins ont également des transactions structurées
        structured_neighbors = 0
        for neighbor in neighbors:
            if isinstance(neighbor, int):  # Vérifier que c'est une entité réelle (pas un nœud externe)
                neighbor_edges = list(G.in_edges(neighbor, data=True)) + list(G.out_edges(neighbor, data=True))
                neighbor_amounts = [e[2]['amount'] for e in neighbor_edges if 'amount' in e[2]]
                
                # Compter les transactions juste en dessous du seuil
                threshold = self.threshold_amount
                margin = 1000
                structured_txns = sum(1 for a in neighbor_amounts if threshold - margin <= a < threshold)
                
                if structured_txns >= 2:
                    structured_neighbors += 1
        
        if structured_neighbors >= 2:
            patterns['structuring_network'] = True
        else:
            patterns['structuring_network'] = False
        
        # 3. Détection de transactions avec pays à risque
        # Vérifier si l'entité a des transactions avec des pays sanctionnés
        suspicious_country_txns = 0
        for edge in in_edges + out_edges:
            edge_data = edge[2]
            country_from = edge_data.get('country_from')
            country_to = edge_data.get('country_to')
            
            if country_from in self.suspicious_countries or country_to in self.suspicious_countries:
                suspicious_country_txns += 1
        
        if suspicious_country_txns > 0:
            patterns['suspicious_country_transactions'] = True
        else:
            patterns['suspicious_country_transactions'] = False
        
        # 4. Détection de circuits fermés (layering)
        # Caractérisé par des fonds qui reviennent à l'entité d'origine après être passés par d'autres entités
        try:
            # Limiter la recherche de cycles pour des raisons de performance
            cycles = []
            for path in nx.all_simple_paths(G, entity_id, entity_id, cutoff=4):
                if len(path) > 2:  # Ignorer les auto-boucles
                    cycles.append(path)
            
            if cycles:
                patterns['layering_detected'] = True
            else:
                patterns['layering_detected'] = False
        except:
            patterns['layering_detected'] = False
        
        return patterns
    
    def calculate_score(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Calcule un score de risque basé sur l'analyse de réseau pour une entité spécifique.
        """
        # Prétraiter les données
        filtered_transactions, filtered_wires = self.preprocess_data(transactions_df, wires_df, entity_df)
        
        # Construire le réseau de transactions
        G = self.build_transaction_network(filtered_transactions, filtered_wires)
        
        # Calculer les métriques de réseau
        metrics = self.calculate_network_metrics(G, entity_id)
        
        # Détecter les schémas suspects
        patterns = self.detect_suspicious_patterns(G, entity_id, entity_df)
        
        # Calculer le score basé sur les métriques et les schémas
        score = 0
        
        # 1. Score basé sur les métriques de réseau
        for metric_name, weight in self.network_metrics.items():
            if metric_name in metrics:
                # Normaliser la métrique entre 0 et 1 si nécessaire
                normalized_metric = min(1.0, metrics[metric_name])
                score += normalized_metric * weight
        
        # 2. Score basé sur les schémas suspects
        if patterns.get('potential_money_mule', False):
            score += 8
        
        if patterns.get('structuring_network', False):
            score += 7
        
        if patterns.get('suspicious_country_transactions', False):
            score += 6
        
        if patterns.get('layering_detected', False):
            score += 9
        
        # Normaliser au max_score
        normalized_score = min(score, self.max_score)
        return normalized_score
    
    def get_network_details(self, entity_id, transactions_df, wires_df, entity_df):
        """
        Retourne les détails de l'analyse de réseau pour une entité spécifique.
        Utile pour expliquer pourquoi une entité a reçu un score élevé.
        """
        # Prétraiter les données
        filtered_transactions, filtered_wires = self.preprocess_data(transactions_df, wires_df, entity_df)
        
        # Construire le réseau de transactions
        G = self.build_transaction_network(filtered_transactions, filtered_wires)
        
        # Calculer les métriques de réseau
        metrics = self.calculate_network_metrics(G, entity_id)
        
        # Détecter les schémas suspects
        patterns = self.detect_suspicious_patterns(G, entity_id, entity_df)
        
        # Combiner les résultats
        details = {
            'metrics': metrics,
            'patterns': patterns,
            'network_size': len(G),
            'direct_connections': G.degree(entity_id) if G.has_node(entity_id) else 0
        }
        
        return details