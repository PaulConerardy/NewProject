import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(G, title):
    """
    Visualise un graphe de transactions avec des couleurs distinctes pour les arêtes entrantes et sortantes.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - title (str) : Titre du graphe.
    """
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Position des nœuds avec espacement

    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Dessiner les arêtes avec des couleurs distinctes
    for edge in G.edges(data=True):
        if G.has_edge(edge[1], edge[0]):  # Vérifie si l'arête inverse existe
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], arrows=True, arrowsize=20, edge_color='green')
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], arrows=True, arrowsize=20, edge_color='red')

    # Dessiner les labels des nœuds et des arêtes
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.axis('off')  # Masquer les axes
    plt.show()
