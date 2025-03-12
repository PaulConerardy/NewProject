import pandas as pd
import numpy as np
import os
import time
from rule_based_detection import RuleBasedDetection
from network_analysis_detection import NetworkAnalysisDetection

def load_data():
    """Charge les données synthétiques à partir des fichiers CSV."""
    # Définir le répertoire de données
    data_dir = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/data"
    
    # Charger les données d'entité
    entity_path = os.path.join(data_dir, "synthetic_entity.csv")
    entity_df = pd.read_csv(entity_path, sep=',')
    
    # Charger les données de transaction
    trx_path = os.path.join(data_dir, "synthetic_trx.csv")
    trx_df = pd.read_csv(trx_path, sep=',')
    
    # Charger les données de virement
    wires_path = os.path.join(data_dir, "synthetic_wires.csv")
    wires_df = pd.read_csv(wires_path, sep=',')
    
    return entity_df, trx_df, wires_df

def analyze_entities(entity_df, trx_df, wires_df, use_network_analysis=True):
    """
    Analyse les entités pour détecter des activités suspectes en utilisant
    à la fois la détection basée sur les règles et l'analyse de réseau.
    """
    # Initialiser les détecteurs
    rule_detector = RuleBasedDetection()
    network_detector = NetworkAnalysisDetection() if use_network_analysis else None
    
    # Stocker les résultats
    results = []
    print(entity_df.columns)
    print(entity_df.head())
    # Analyser chaque entité unique
    unique_entities = entity_df['party_key'].unique()
    
    print(f"Analyse de {len(unique_entities)} entités...")
    start_time = time.time()
    
    for i, entity_id in enumerate(unique_entities):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            estimated_total = elapsed / (i + 1) * len(unique_entities)
            remaining = estimated_total - elapsed
            print(f"Progression: {i}/{len(unique_entities)} entités analysées ({elapsed:.1f}s écoulées, ~{remaining:.1f}s restantes)")
        
        # Calculer le score basé sur les règles
        rule_score = rule_detector.calculate_score(entity_id, trx_df, wires_df, entity_df)
        
        # Calculer le score basé sur l'analyse de réseau (si activée)
        network_score = 0
        if use_network_analysis and network_detector:
            network_score = network_detector.calculate_score(entity_id, trx_df, wires_df, entity_df)
        
        # Combiner les scores (moyenne pondérée)
        if use_network_analysis:
            combined_score = 0.6 * rule_score + 0.4 * network_score
        else:
            combined_score = rule_score
        
        # Si le score est supérieur à zéro, obtenir les détails
        rule_details = {}
        network_details = {}
        
        if rule_score > 0:
            rule_details = rule_detector.get_rule_details(entity_id, trx_df, wires_df, entity_df)
        
        if use_network_analysis and network_score > 0:
            network_details = network_detector.get_network_details(entity_id, trx_df, wires_df, entity_df)
        
        # Obtenir les informations de l'entité
        entity_info = entity_df[entity_df['party_key'] == entity_id].iloc[0]
        
        # Déterminer si l'entité est suspecte (score > 15)
        is_suspicious = combined_score >= 15
        
        # Stocker les résultats
        results.append({
            'entity_id': entity_id,
            'entity_name': entity_info['party_name'] if 'party_name' in entity_info else f"Entity {entity_id}",
            'entity_type': entity_info['account_type_desc'],
            'rule_score': rule_score,
            'network_score': network_score,
            'combined_score': combined_score,
            'is_suspicious': is_suspicious,
            'triggered_rules': rule_details,
            'network_patterns': network_details.get('patterns', {}) if network_details else {}
        })
    
    print(f"Analyse terminée en {time.time() - start_time:.2f} secondes")
    return results

def generate_report(results, use_network_analysis=True):
    """Génère un rapport détaillé des entités suspectes."""
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame([
        {
            'entity_id': r['entity_id'],
            'entity_name': r['entity_name'],
            'entity_type': r['entity_type'],
            'rule_score': r['rule_score'],
            'network_score': r['network_score'] if use_network_analysis else 0,
            'combined_score': r['combined_score'],
            'is_suspicious': r['is_suspicious'],
            'triggered_rules_count': len(r['triggered_rules']),
            'network_patterns_count': sum(1 for v in r['network_patterns'].values() if v) if use_network_analysis else 0
        } for r in results
    ])
    
    # Filtrer les entités suspectes
    suspicious_df = results_df[results_df['is_suspicious']]
    
    print("\n=== RAPPORT DE DÉTECTION COMBINÉE ===")
    print(f"Total des entités analysées: {len(results_df)}")
    print(f"Entités suspectes détectées: {len(suspicious_df)}")
    
    if len(suspicious_df) > 0:
        print("\nTop 10 des entités les plus suspectes:")
        top_suspicious = suspicious_df.sort_values('combined_score', ascending=False).head(10)
        for _, row in top_suspicious.iterrows():
            print(f"ID: {row['entity_id']}, Nom: {row['entity_name']}, Type: {row['entity_type']}, Score: {row['combined_score']:.2f}")
            
            # Afficher les règles déclenchées pour cette entité
            entity_result = next(r for r in results if r['entity_id'] == row['entity_id'])
            rule_descriptions = RuleBasedDetection().get_rule_descriptions()
            
            print("  Règles déclenchées:")
            for rule, score in entity_result['triggered_rules'].items():
                print(f"  - {rule_descriptions[rule]} (Score: {score})")
            
            # Afficher les schémas de réseau détectés
            if use_network_analysis:
                print("  Schémas de réseau détectés:")
                network_patterns = entity_result['network_patterns']
                pattern_descriptions = {
                    'potential_money_mule': "Potentielle mule d'argent",
                    'structuring_network': "Réseau de structuration",
                    'suspicious_country_transactions': "Transactions avec pays à risque",
                    'layering_detected': "Circuit fermé de transactions (layering)"
                }
                
                for pattern, detected in network_patterns.items():
                    if detected and pattern in pattern_descriptions:
                        print(f"  - {pattern_descriptions[pattern]}")
            
            print()
    
    # Enregistrer les résultats complets dans un CSV
    output_path = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/new/combined_detection_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Résultats complets enregistrés dans {output_path}")
    
    # Enregistrer les détails des entités suspectes dans un CSV séparé
    if len(suspicious_df) > 0:
        suspicious_output_path = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/new/suspicious_entities_combined.csv"
        suspicious_df.to_csv(suspicious_output_path, index=False)
        print(f"Détails des entités suspectes enregistrés dans {suspicious_output_path}")
    
    return results_df, suspicious_df

def generate_network_visualization(entity_id, transactions_df, wires_df, entity_df, output_dir):
    """
    Génère une visualisation du réseau de transactions pour une entité spécifique.
    Utile pour l'analyse manuelle des schémas suspects.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Initialiser le détecteur de réseau
        network_detector = NetworkAnalysisDetection()
        
        # Prétraiter les données
        filtered_transactions, filtered_wires = network_detector.preprocess_data(transactions_df, wires_df, entity_df)
        
        # Construire le réseau de transactions
        G = network_detector.build_transaction_network(filtered_transactions, filtered_wires)
        
        # Extraire le sous-graphe pertinent pour l'entité (voisinage à 2 sauts)
        if G.has_node(entity_id):
            neighbors = list(nx.neighbors(G, entity_id))
            neighbors.extend([n for nbr in neighbors for n in nx.neighbors(G, nbr)])
            neighbors = list(set(neighbors))  # Éliminer les doublons
            neighbors.append(entity_id)
            subgraph = G.subgraph(neighbors).copy()
            
            # Créer la figure
            plt.figure(figsize=(12, 10))
            
            # Définir les positions des nœuds
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Définir les couleurs des nœuds
            node_colors = []
            for node in subgraph.nodes():
                if node == entity_id:
                    node_colors.append('red')  # Entité cible en rouge
                elif isinstance(node, int):
                    node_colors.append('blue')  # Autres entités en bleu
                elif 'external' in str(node):
                    node_colors.append('green')  # Entités externes en vert
                else:
                    node_colors.append('gray')  # Nœuds inconnus en gris
            
            # Dessiner les nœuds
            nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color=node_colors, alpha=0.8)
            
            # Dessiner les arêtes
            edge_colors = []
            edge_widths = []
            for u, v, data in subgraph.edges(data=True):
                amount = data.get('amount', 0)
                if amount > 10000:
                    edge_colors.append('red')
                    edge_widths.append(2.0)
                elif amount > 5000:
                    edge_colors.append('orange')
                    edge_widths.append(1.5)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1.0)
            
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, arrows=True, arrowsize=15)
            
            # Dessiner les étiquettes des nœuds
            node_labels = {}
            for node in subgraph.nodes():
                if node == entity_id:
                    # Obtenir le nom de l'entité
                    entity_info = entity_df[entity_df['party_key'] == entity_id]
                    if not entity_info.empty:
                        entity_name = entity_info.iloc[0]['party_name']
                        node_labels[node] = f"{entity_name}\n(ID: {entity_id})"
                    else:
                        node_labels[node] = f"ID: {entity_id}"
                elif isinstance(node, int):
                    # Obtenir le nom de l'entité si disponible
                    entity_info = entity_df[entity_df['party_key'] == node]
                    if not entity_info.empty:
                        entity_name = entity_info.iloc[0]['party_name']
                        node_labels[node] = f"{entity_name[:10]}...\n(ID: {node})"
                    else:
                        node_labels[node] = f"ID: {node}"
                elif 'external' in str(node):
                    # Extraire le nom de l'entité externe
                    external_name = str(node).replace('external_', '')
                    node_labels[node] = f"{external_name[:15]}..."
                else:
                    node_labels[node] = ""
            
            nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8, font_weight='bold')
            
            # Ajouter un titre
            entity_info = entity_df[entity_df['party_key'] == entity_id]
            if not entity_info.empty:
                entity_name = entity_info.iloc[0]['party_name']
                plt.title(f"Réseau de transactions pour {entity_name} (ID: {entity_id})")
            else:
                plt.title(f"Réseau de transactions pour l'entité ID: {entity_id}")
            
            # Enregistrer la figure
            output_file = os.path.join(output_dir, f"network_entity_{entity_id}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualisation du réseau enregistrée dans {output_file}")
            return output_file
        else:
            print(f"L'entité {entity_id} n'existe pas dans le réseau")
            return None
    except Exception as e:
        print(f"Erreur lors de la génération de la visualisation: {e}")
        return None

def main():
    """Fonction principale pour exécuter le système de détection combiné."""
    print("Chargement des données...")
    entity_df, trx_df, wires_df = load_data()
    
    print(f"Données chargées: {len(entity_df)} entités, {len(trx_df)} transactions, {len(wires_df)} virements")
    
    # Demander à l'utilisateur s'il souhaite utiliser l'analyse de réseau
    use_network = input("Utiliser l'analyse de réseau en plus de la détection basée sur les règles? (o/n): ").lower() == 'o'
    
    print("Analyse des entités...")
    results = analyze_entities(entity_df, trx_df, wires_df, use_network_analysis=use_network)
    
    print("Génération du rapport...")
    results_df, suspicious_df = generate_report(results, use_network_analysis=use_network)
    
    # Demander à l'utilisateur s'il souhaite générer des visualisations pour les entités suspectes
    if use_network and len(suspicious_df) > 0:
        generate_viz = input("Générer des visualisations de réseau pour les entités suspectes? (o/n): ").lower() == 'o'
        
        if generate_viz:
            # Créer le répertoire de sortie s'il n'existe pas
            output_dir = "/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/new/network_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # Générer des visualisations pour les 5 entités les plus suspectes
            top_suspicious = suspicious_df.sort_values('combined_score', ascending=False).head(5)
            
            for _, row in top_suspicious.iterrows():
                entity_id = row['entity_id']
                print(f"Génération de la visualisation pour l'entité {entity_id}...")
                generate_network_visualization(entity_id, trx_df, wires_df, entity_df, output_dir)
    
    print("\nAnalyse terminée.")

if __name__ == "__main__":
    main()