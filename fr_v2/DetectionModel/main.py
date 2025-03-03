from data_generator import TransactionDataGenerator
from network_analysis import NetworkAnalysis
import matplotlib.pyplot as plt
import networkx as nx

def visualize_network(G, communities):
    """Fonction d'aide pour visualiser le réseau"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Dessiner les nœuds
    colors = ['r', 'g', 'b', 'y', 'c']
    for idx, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community),
                             node_color=colors[idx % len(colors)],
                             node_size=100, alpha=0.6)
    
    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("Réseau de Transactions avec Communautés")
    plt.show()

def main():
    # Générer des données synthétiques
    generator = TransactionDataGenerator(num_entities=100, num_transactions=1000)
    entity_data = generator.generate_entity_data()
    transaction_data = generator.generate_transaction_data()
    
    print("Données synthétiques générées :")
    print(f"Nombre d'entités : {len(entity_data)}")
    print(f"Nombre de transactions : {len(transaction_data)}")
    
    # Créer une instance d'analyse de réseau
    network = NetworkAnalysis()
    
    # Construire et analyser le réseau
    G = network.build_transaction_network(transaction_data, entity_data)
    
    # Calculer les métriques de réseau
    metrics = network.calculate_network_metrics(G)
    print("\nRésumé des métriques de réseau :")
    print(metrics.describe())
    
    # Détecter les communautés
    communities, community_mapping = network.detect_communities(G)
    
    # Analyser les flux des communautés
    flow_metrics = network.analyze_community_flows(G, communities, transaction_data)
    print("\nMétriques de flux des communautés :")
    print(flow_metrics.head())
    
    # Identifier les entités de pont
    bridge_entities = network.identify_bridge_entities(G, communities)
    print(f"\nIdentifié {len(bridge_entities)} entités de pont :")
    print(bridge_entities[:5])  # Afficher les 5 premières entités de pont
    
    # Calculer les métriques temporelles pour une entité exemple
    sample_entity = entity_data['entity_id'].iloc[0]
    temporal_metrics = network.calculate_temporal_metrics(transaction_data, sample_entity)
    print(f"\nMétriques temporelles pour l'entité {sample_entity} :")
    print(temporal_metrics.head())
    
    # Visualiser le réseau
    visualize_network(G, communities)

if __name__ == "__main__":
    main()