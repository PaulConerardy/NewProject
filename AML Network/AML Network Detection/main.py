import networkx as nx
from config_loader import load_config, load_data
from visualization import plot_graph
from graph_utils import create_graph, get_depth, breadth_first_search, get_transaction_traffic_per_node, get_normalized_edge_weights
from detection import get_suspicious_transactions, get_suspicious_traffic, get_gamma, sort_suspicious_transactions, filter_transactions

def main(config_file='config.json', data_file='small_transactions.csv'):
    """
    Fonction principale pour exécuter l'analyse des transactions suspectes.

    Paramètres :
    - config_file (str) : Chemin vers le fichier de configuration.
    - data_file (str) : Chemin vers le fichier de données.
    """
    config = load_config(config_file)
    df = load_data(data_file)
    df_after_cutoff, filtered_by_amount_per_node = filter_transactions(df, config['money_cut_off'])

    # Construction des arêtes du graphe à partir des transactions
    graph_edge_constructor = df_after_cutoff.groupby('date')[
        ['SENDER', 'RECEIVER', 'AMOUNT']].apply(lambda x: x.values.tolist()).to_dict()

    # Initialisation des structures de données pour stocker les résultats
    all_normalized_transactions, transaction_volume, betweenness_dict, closeness_dict = [], {}, {}, {}

    # Boucle principale pour traiter les transactions par jour
    for date, all_transactions_per_time in zip(graph_edge_constructor.keys(), graph_edge_constructor.values()):
        G = create_graph([trans for trans in all_transactions_per_time])
        max_subgraph_size, visited_subgraphs = 0, []

        for transaction in all_transactions_per_time:
            depth = get_depth(G, transaction[0])

            if depth > 1:
                connected_nodes = breadth_first_search(G, transaction[0])

                if max_subgraph_size < len(connected_nodes) or not set(connected_nodes) <= set(visited_subgraphs):
                    max_subgraph_size = len(connected_nodes)
                    visited_subgraphs += connected_nodes
                    sub_G = G.subgraph(connected_nodes)
                    betweenness_nodes = nx.betweenness_centrality(sub_G, normalized=True)
                    closeness_nodes = nx.closeness_centrality(sub_G)

                    normalized_transactions, transaction_volume = get_normalized_edge_weights(G, date, connected_nodes, transaction_volume, filtered_by_amount_per_node)

                    if normalized_transactions not in all_normalized_transactions:
                        all_normalized_transactions.append(normalized_transactions)

    # Identifier les transactions suspectes
    suspicious_transactions = get_suspicious_transactions(all_normalized_transactions, config['commission_cut_off'])

    # Normaliser le trafic suspect
    suspicious_traffic = get_suspicious_traffic(transaction_volume)

    # Calculer le paramètre gamma
    gamma_nodes, gamma_dict = get_gamma(suspicious_traffic, betweenness_dict, closeness_dict, suspicious_transactions)

    # Classer les transactions suspectes
    suspicious_transactions_by_line = sort_suspicious_transactions(suspicious_transactions, gamma_dict)

    # Regrouper les transactions par chaîne avant de les visualiser
    transaction_chains = []
    current_chain = []
    current_date = None

    for transaction in suspicious_transactions_by_line:
        if current_date is None:
            current_date = transaction[0]

        if transaction[0] == current_date:
            current_chain.append(transaction)
        else:
            transaction_chains.append(current_chain)
            current_chain = [transaction]
            current_date = transaction[0]

    if current_chain:
        transaction_chains.append(current_chain)

    # Visualisation des graphes des transactions les plus suspectes
    top_n_suspicious_transactions = transaction_chains[:config['top_n_suspicious']]
    for i, transaction_chain in enumerate(top_n_suspicious_transactions):
        edges = [(trans[1], trans[2], trans[3]) for trans in transaction_chain]
        G = create_graph(edges)
        plot_graph(G, f"Top {i+1} des transactions suspectes")

if __name__ == "__main__":
    main()
