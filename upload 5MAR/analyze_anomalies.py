from network_analysis import NetworkAnalysis
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

def visualize_network(G, communities, title="Réseau de Transactions avec Communautés", output_file=None):
    """Fonction auxiliaire pour visualiser le réseau"""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Dessiner les nœuds
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'brown', 'pink']
    for idx, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community),
                             node_color=colors[idx % len(colors)],
                             node_size=150, alpha=0.7)
    
    # Dessiner les arêtes avec une largeur proportionnelle au montant de la transaction
    edge_widths = []
    for u, v, data in G.edges(data=True):
        edge_widths.append(0.1 + 0.9 * min(data['amount'] / 10000, 5))
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths)
    
    # Ajouter des étiquettes aux nœuds
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    
    plt.title(title)
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def identify_suspicious_transactions(transaction_data, G, bridge_entities, communities, metrics):
    """
    Identifier les transactions suspectes basées sur les résultats de l'analyse du réseau
    
    Paramètres:
    -----------
    transaction_data : DataFrame
        Données des transactions
    G : NetworkX Graph
        Réseau de transactions
    bridge_entities : list
        Liste des entités pont
    communities : list
        Liste des communautés
    metrics : DataFrame
        Métriques du réseau des entités
    
    Retourne:
    --------
    DataFrame
        Transactions suspectes avec scores de risque
    """
    # Créer une copie des données de transaction
    suspicious_txns = transaction_data.copy()
    
    # Initialiser le score de suspicion
    suspicious_txns['suspicion_score'] = 0.0
    suspicious_txns['suspicion_reason'] = ''
    
    # 1. Transactions impliquant des entités pont (risque élevé)
    bridge_mask = (suspicious_txns['sender_id'].isin(bridge_entities)) | \
                 (suspicious_txns['receiver_id'].isin(bridge_entities))
    suspicious_txns.loc[bridge_mask, 'suspicion_score'] += 3.0
    suspicious_txns.loc[bridge_mask, 'suspicion_reason'] += 'Entité pont impliquée; '
    
    # 2. Transactions entre différentes communautés (risque moyen)
    community_dict = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_dict[node] = i
    
    for idx, row in suspicious_txns.iterrows():
        sender = row['sender_id']
        receiver = row['receiver_id']
        
        # Vérifier si l'expéditeur et le destinataire sont dans différentes communautés
        if sender in community_dict and receiver in community_dict:
            if community_dict[sender] != community_dict[receiver]:
                suspicious_txns.loc[idx, 'suspicion_score'] += 2.0
                suspicious_txns.loc[idx, 'suspicion_reason'] += 'Transaction inter-communautaire; '
    
    # 3. Transactions avec des entités à forte centralité d'intermédiarité (risque moyen-élevé)
    if 'betweenness_centrality' in metrics.columns:
        high_betweenness = metrics[metrics['betweenness_centrality'] > 
                                  metrics['betweenness_centrality'].quantile(0.9)]['entity_id'].tolist()
        
        betweenness_mask = (suspicious_txns['sender_id'].isin(high_betweenness)) | \
                          (suspicious_txns['receiver_id'].isin(high_betweenness))
        suspicious_txns.loc[betweenness_mask, 'suspicion_score'] += 2.5
        suspicious_txns.loc[betweenness_mask, 'suspicion_reason'] += 'Entité à forte centralité impliquée; '
    
    # 4. Transactions avec des montants inhabituellement élevés (risque moyen)
    amount_threshold = suspicious_txns['amount'].quantile(0.95)
    amount_mask = suspicious_txns['amount'] > amount_threshold
    suspicious_txns.loc[amount_mask, 'suspicion_score'] += 2.0
    suspicious_txns.loc[amount_mask, 'suspicion_reason'] += 'Montant inhabituellement élevé; '
    
    # 5. Transactions avec des flux équilibrés (potentiel système bancaire souterrain)
    # Rechercher des transactions avec des montants similaires entre les mêmes entités mais dans des directions opposées
    for idx, row in suspicious_txns.iterrows():
        sender = row['sender_id']
        receiver = row['receiver_id']
        amount = row['amount']
        timestamp = pd.to_datetime(row['timestamp'])
        
        # Rechercher des transactions inverses avec des montants similaires dans les 7 jours
        reverse_txns = suspicious_txns[
            (suspicious_txns['sender_id'] == receiver) & 
            (suspicious_txns['receiver_id'] == sender) &
            (abs(suspicious_txns['amount'] - amount) / amount < 0.1)  # Dans une marge de 10% du montant
        ]
        
        if not reverse_txns.empty:
            for r_idx, r_row in reverse_txns.iterrows():
                r_timestamp = pd.to_datetime(r_row['timestamp'])
                time_diff = abs((timestamp - r_timestamp).total_seconds() / (24*3600))  # jours
                
                if time_diff <= 7:  # Dans les 7 jours
                    suspicious_txns.loc[idx, 'suspicion_score'] += 4.0
                    suspicious_txns.loc[idx, 'suspicion_reason'] += 'Flux équilibré (potentiel système bancaire souterrain); '
                    suspicious_txns.loc[r_idx, 'suspicion_score'] += 4.0
                    suspicious_txns.loc[r_idx, 'suspicion_reason'] += 'Flux équilibré (potentiel système bancaire souterrain); '
    
    # Filtrer pour n'inclure que les transactions suspectes (score > 0)
    suspicious_txns = suspicious_txns[suspicious_txns['suspicion_score'] > 0]
    
    # Trier par score de suspicion (décroissant)
    suspicious_txns = suspicious_txns.sort_values('suspicion_score', ascending=False)
    
    # 6. Transactions multiples sous le seuil de déclaration (structuration)
    # Identifier les transactions sous 10 000$
    threshold = 10000
    under_threshold_mask = suspicious_txns['amount'] < threshold
    suspicious_txns.loc[under_threshold_mask, 'suspicion_score'] += 1.0
    suspicious_txns.loc[under_threshold_mask, 'suspicion_reason'] += 'Montant sous le seuil de déclaration; '
    
    # 7. Transactions internationales
    # Créer un dictionnaire pour mapper les entités à leurs pays (si disponible)
    if 'country' in G.nodes[list(G.nodes())[0]]:
        entity_countries = {n: G.nodes[n]['country'] for n in G.nodes()}
        
        for idx, row in suspicious_txns.iterrows():
            sender = row['sender_id']
            receiver = row['receiver_id']
            
            if sender in entity_countries and receiver in entity_countries:
                if entity_countries[sender] != entity_countries[receiver]:
                    suspicious_txns.loc[idx, 'suspicion_score'] += 2.0
                    suspicious_txns.loc[idx, 'suspicion_reason'] += 'Transaction internationale; '
    
    # 8. Transactions avec des entités ayant un grand volume de dépôts de tiers
    # Cette partie nécessiterait une analyse préalable des modèles de dépôt
    # Nous utilisons ici une heuristique basée sur le degré entrant
    high_in_degree = metrics[metrics['in_degree'] > metrics['in_degree'].quantile(0.9)]['entity_id'].tolist()
    high_in_mask = suspicious_txns['receiver_id'].isin(high_in_degree)
    suspicious_txns.loc[high_in_mask, 'suspicion_score'] += 1.5
    suspicious_txns.loc[high_in_mask, 'suspicion_reason'] += 'Entité avec volume élevé de dépôts; '
    
    return suspicious_txns

def analyze_anomalies():
    """Analyser le réseau de transactions des entités anormales"""
    # Charger les données d'anomalie
    anomaly_dir = '/Users/paulconerardy/Documents/Trae/ESM3/anomaly_data'
    entity_file = os.path.join(anomaly_dir, 'anomaly_entities.csv')
    transaction_file = os.path.join(anomaly_dir, 'anomaly_transactions.csv')
    
    if not os.path.exists(entity_file) or not os.path.exists(transaction_file):
        print("Fichiers de données d'anomalie non trouvés. Veuillez d'abord exécuter generate_anomaly_transactions.py.")
        return
    
    entity_data = pd.read_csv(entity_file)
    transaction_data = pd.read_csv(transaction_file)
    
    # Convertir le timestamp en format datetime
    transaction_data['timestamp'] = pd.to_datetime(transaction_data['timestamp'])
    
    print("Données d'anomalie chargées:")
    print(f"Nombre d'entités: {len(entity_data)}")
    print(f"Nombre de transactions: {len(transaction_data)}")
    
    # Créer une instance d'analyse de réseau
    network = NetworkAnalysis()
    
    # Construire et analyser le réseau
    G = network.build_transaction_network(transaction_data, entity_data)
    
    # Calculer les métriques du réseau
    metrics = network.calculate_network_metrics(G)
    print("\nRésumé des métriques du réseau:")
    print(metrics.describe())
    
    # Sauvegarder les métriques en CSV
    metrics.to_csv('/Users/paulconerardy/Documents/Trae/ESM3/anomaly_network_metrics.csv', index=False)
    
    # Détecter les communautés
    communities, community_mapping = network.detect_communities(G)
    
    # Analyser les flux des communautés
    flow_metrics = network.analyze_community_flows(G, communities, transaction_data)
    print("\nMétriques des flux des communautés:")
    print(flow_metrics.head())
    
    # Sauvegarder les métriques de flux en CSV
    flow_metrics.to_csv('/Users/paulconerardy/Documents/Trae/ESM3/anomaly_community_flows.csv', index=False)
    
    # Identifier les entités pont
    bridge_entities = network.identify_bridge_entities(G, communities)
    print(f"\nIdentifié {len(bridge_entities)} entités pont:")
    print(bridge_entities[:5])  # Afficher les 5 premières entités pont
    
    # Sauvegarder les entités pont en CSV
    pd.DataFrame({'entity_id': bridge_entities}).to_csv(
        '/Users/paulconerardy/Documents/Trae/ESM3/anomaly_bridge_entities.csv', index=False
    )
    
    # Calculer les métriques temporelles pour chaque entité anormale
    temporal_metrics_list = []
    
    for entity_id in entity_data['entity_id']:
        try:
            # Créer une copie des données de transaction pour cette entité
            entity_txns = transaction_data[
                (transaction_data['sender_id'] == entity_id) | 
                (transaction_data['receiver_id'] == entity_id)
            ].copy()
            
            # S'assurer que le timestamp est en datetime
            entity_txns['timestamp'] = pd.to_datetime(entity_txns['timestamp'])
            
            # Calculer les métriques temporelles
            temp_metrics = network.calculate_temporal_metrics(entity_txns, entity_id)
            temp_metrics['entity_id'] = entity_id
            temporal_metrics_list.append(temp_metrics)
        except Exception as e:
            print(f"Erreur lors du calcul des métriques temporelles pour l'entité {entity_id}: {e}")
    
    if temporal_metrics_list:
        all_temporal_metrics = pd.concat(temporal_metrics_list)
        all_temporal_metrics.to_csv('/Users/paulconerardy/Documents/Trae/ESM3/anomaly_temporal_metrics.csv', index=False)
    
    # Identifier les transactions suspectes
    print("\nIdentification des transactions suspectes...")
    suspicious_txns = identify_suspicious_transactions(
        transaction_data, G, bridge_entities, communities, metrics
    )
    
    # Sauvegarder les transactions suspectes en CSV
    suspicious_file = '/Users/paulconerardy/Documents/Trae/ESM3/suspicious_transactions.csv'
    suspicious_txns.to_csv(suspicious_file, index=False)
    print(f"Identifié {len(suspicious_txns)} transactions suspectes et sauvegardé dans {suspicious_file}")
    
    # Imprimer le résumé des transactions suspectes
    print("\nRésumé des transactions suspectes:")
    print(f"Total des transactions suspectes: {len(suspicious_txns)}")
    print(f"Score de suspicion moyen: {suspicious_txns['suspicion_score'].mean():.2f}")
    print("\n5 transactions les plus suspectes:")
    print(suspicious_txns[['transaction_id', 'sender_id', 'receiver_id', 'amount', 
                          'suspicion_score', 'suspicion_reason']].head(5))
    
    # Visualiser le réseau
    visualize_network(G, communities, 
                     title="Réseau de Transactions des Entités Anormales",
                     output_file='/Users/paulconerardy/Documents/Trae/ESM3/anomaly_network.png')
    
    # Créer un sous-graphe des principales entités pont pour une analyse détaillée
    if bridge_entities:
        top_bridges = bridge_entities[:min(5, len(bridge_entities))]
        bridge_subgraph = G.subgraph(top_bridges + 
                                    [n for b in top_bridges for n in G.neighbors(b)])
        
        # Détecter les communautés dans le sous-graphe
        bridge_communities = list(nx.connected_components(bridge_subgraph.to_undirected()))
        
        # Visualiser le sous-graphe des entités pont
        visualize_network(bridge_subgraph, bridge_communities,
                         title="Réseau de Transactions des Principales Entités Pont",
                         output_file='/Users/paulconerardy/Documents/Trae/ESM3/bridge_entity_network.png')
    
    print("\nAnalyse terminée. Résultats sauvegardés en fichiers CSV et visualisations sauvegardées en fichiers PNG.")

if __name__ == "__main__":
    analyze_anomalies()