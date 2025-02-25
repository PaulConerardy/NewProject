import networkx as nx

def create_graph(edges):
    """
    Crée un graphe dirigé multi-arêtes à partir d'une liste d'arêtes.

    Paramètres :
    - edges (list) : Liste des arêtes avec leurs poids.

    Retourne :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    """
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(edges)
    return G

def get_depth(G, node):
    """
    Calcule la profondeur d'un nœud dans le graphe.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - node (any) : Nœud dont on veut calculer la profondeur.

    Retourne :
    - int : Profondeur maximale du nœud.
    """
    return max(nx.single_source_shortest_path_length(G, node).values())

def breadth_first_search(G, node):
    """
    Effectue une recherche en largeur (BFS) à partir d'un nœud donné.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - node (any) : Nœud de départ pour la recherche.

    Retourne :
    - list : Liste des nœuds connectés dans l'ordre BFS.
    """
    return list(nx.shortest_path(G, source=node))

def get_transaction_traffic_per_node(G, time, node, transaction_volume):
    """
    Calcule le paramètre de trafic d'un nœud donné.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - time (int) : Jour ordonné.
    - node (any) : Nœud dont on veut calculer le trafic.
    - transaction_volume (dict) : Dictionnaire du volume de transactions.

    Retourne :
    - dict : Dictionnaire mis à jour du volume de transactions.
    """
    if (time, node) not in transaction_volume:
        transaction_volume[(time, node)] = (G.in_degree(node) + G.out_degree(node))
    else:
        transaction_volume[(time, node)] += (G.in_degree(node) + G.out_degree(node))
    return transaction_volume

def get_normalized_edge_weights(G, time, connected_nodes, transaction_volume, filtered_by_amount_per_node):
    """
    Normalise les poids des arêtes par le montant initial envoyé par l'expéditeur.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - time (int) : Jour ordonné.
    - connected_nodes (list) : Liste des nœuds connectés.
    - transaction_volume (dict) : Dictionnaire du volume de transactions.
    - filtered_by_amount_per_node (dict) : Dictionnaire des montants filtrés par nœud.

    Retourne :
    - tuple : Liste des transactions normalisées et dictionnaire du volume de transactions.
    """
    normalized_transactions = []
    for node in connected_nodes:
        neighbors = G.neighbors(node)
        transaction_volume = get_transaction_traffic_per_node(G, time, node, transaction_volume)
        
        betweenness_dict[(time, node)] = betweenness_nodes[node]
        closeness_dict[(time, node)] = closeness_nodes[node]

        for neighbor in neighbors:
            for k in range(len(G.get_edge_data(node, neighbor))):
                edge_amount = G.get_edge_data(node, neighbor)[k]['weight']
                initial_sent_amount = filtered_by_amount_per_node[tuple([time, connected_nodes[0]])]
                normalized_transactions.append([time, node, neighbor, edge_amount, edge_amount / initial_sent_amount])

    return normalized_transactions, transaction_volume
