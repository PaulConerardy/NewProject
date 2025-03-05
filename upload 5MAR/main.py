from data_generator import TransactionDataGenerator
from network_analysis import NetworkAnalysis
from detection_patterns import DetectionPatterns
from alert_system import AlertSystem
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

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
    
    # Appliquer les schémas de détection
    detection = DetectionPatterns()
    
    # 1. Détecter les schémas de structuration
    print("\nDétection des schémas de structuration...")
    structuring_entities = detection.detect_structuring(transaction_data)
    print(f"Identifié {len(structuring_entities)} entités avec des schémas de structuration")
    
    # 2. Détecter les potentielles mules financières
    print("\nDétection des potentielles mules financières...")
    money_mules = detection.detect_money_mules(transaction_data, entity_data)
    if not money_mules.empty:
        print(f"Identifié {len(money_mules)} potentielles mules financières")
        print(money_mules.head())
    
    # 3. Détecter les dépôts structurés
    print("\nDétection des dépôts structurés...")
    structured_deposits = detection.detect_structured_deposits(transaction_data)
    if not structured_deposits.empty:
        print(f"Identifié {len(structured_deposits)} cas de dépôts structurés")
        print(structured_deposits.head())
    
    # 4. Détecter les systèmes bancaires clandestins
    print("\nDétection des systèmes bancaires clandestins...")
    underground_banking = detection.detect_underground_banking(transaction_data, entity_data)
    if not underground_banking.empty:
        print(f"Identifié {len(underground_banking)} potentiels systèmes bancaires clandestins")
        print(underground_banking.head())
    
    # 5. Détecter les schémas de schtroumpfage
    print("\nDétection des schémas de schtroumpfage...")
    smurfing_entities = detection.detect_smurfing(transaction_data)
    print(f"Identifié {len(smurfing_entities)} entités avec des schémas de schtroumpfage")
    
    # 6. Détecter les mouvements rapides de fonds
    print("\nDétection des mouvements rapides de fonds...")
    rapid_movement_entities = detection.detect_rapid_movement(transaction_data, entity_data, G)
    print(f"Identifié {len(rapid_movement_entities)} entités impliquées dans des mouvements rapides de fonds")
    
    # Regrouper les résultats de l'analyse de réseau
    network_results = {
        'metrics': metrics,
        'bridge_entities': bridge_entities,
        'communities': communities,
        'community_mapping': community_mapping,
        'flow_metrics': flow_metrics
    }
    
    # Regrouper les résultats de la détection de schémas
    pattern_results = {
        'structuring_entities': structuring_entities,
        'smurfing_entities': smurfing_entities,
        'rapid_movement_entities': rapid_movement_entities,
        'money_mules': money_mules,
        'structured_deposits': structured_deposits,
        'underground_banking': underground_banking
    }
    
    # Générer des alertes basées sur les résultats de détection
    print("\nGénération des alertes...")
    alert_system = AlertSystem(alert_threshold=50)
    alerts = alert_system.generate_alerts(entity_data, transaction_data, network_results, pattern_results)
    
    if not alerts.empty:
        print(f"Généré {len(alerts)} alertes")
        print("\nTop 5 alertes par score de risque :")
        print(alerts.head())
        
        # Afficher le résumé des alertes
        alert_summary = alert_system.get_alert_summary()
        print("\nRésumé des alertes :")
        print(f"Total des alertes : {alert_summary['total_alerts']}")
        print(f"Priorité élevée : {alert_summary['priority']['high']}")
        print(f"Priorité moyenne : {alert_summary['priority']['medium']}")
        print(f"Priorité faible : {alert_summary['priority']['low']}")
    else:
        print("Aucune alerte générée")
    
    # Visualiser le réseau
    visualize_network(G, communities)
    
    # Sauvegarder les résultats
    results_dir = '/Users/paulconerardy/Documents/Trae/fr_v2/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Sauvegarder les métriques du réseau
    metrics.to_csv(os.path.join(results_dir, 'network_metrics.csv'), index=False)
    
    # Sauvegarder les alertes
    if not alerts.empty:
        alerts.to_csv(os.path.join(results_dir, 'alerts.csv'), index=False)
        
        # Sauvegarder les alertes par priorité
        high_priority = alerts[alerts['priority'] == 'Élevée']
        if not high_priority.empty:
            high_priority.to_csv(os.path.join(results_dir, 'high_priority_alerts.csv'), index=False)
        
        medium_priority = alerts[alerts['priority'] == 'Moyenne']
        if not medium_priority.empty:
            medium_priority.to_csv(os.path.join(results_dir, 'medium_priority_alerts.csv'), index=False)
    
    print(f"\nRésultats sauvegardés dans {results_dir}")

if __name__ == "__main__":
    main()