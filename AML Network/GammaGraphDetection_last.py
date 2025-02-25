import pandas as pd
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# Paramètres définis par l'utilisateur
money_cut_off = 10000.  # Seuil de montant pour filtrer les transactions
commission_cut_off = 0.9  # Seuil de commission pour détecter les transactions suspectes
group_by_time = 1  # Intervalle de temps en jours pour regrouper les transactions

# Chargement des données
df = pd.read_csv('small_transactions.csv', sep='|')
df = df[df['TIMESTAMP'] != '2006-02-29']  # Suppression des dates invalides

# Conversion des timestamps en objets datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df['date'] = df['TIMESTAMP'].apply(lambda x: x.toordinal())  # Conversion en jours ordonnés

# Agrégation des transactions par date et expéditeur
agg = df.groupby(['date', 'SENDER'])['AMOUNT'].sum()
agg = agg[agg > money_cut_off]  # Filtrage des transactions supérieures au seuil

# Réinitialisation de l'index pour obtenir un DataFrame plat
agg_flat = agg.reset_index()

# Fusion des données agrégées avec les données originales
df_after_cutoff = pd.merge(agg_flat[['date', 'SENDER']], df, how='left')

# Dictionnaire des montants totaux par expéditeur et par jour
filtered_by_amount_per_node = agg.to_dict()

# Construction des arêtes du graphe à partir des transactions
graph_edge_constructor = df_after_cutoff.groupby('date')[
    ['SENDER', 'RECEIVER', 'AMOUNT']].apply(lambda x: x.values.tolist()).to_dict()

# Construction des transactions filtrées par jour et par transaction
filtered_transactions = df_after_cutoff.groupby(['date', 'TRANSACTION'])[
    ['SENDER', 'RECEIVER', 'AMOUNT']].apply(lambda x: x.values.tolist()).to_dict()

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

def get_tansaction_traffic_per_node(G, time, node, transaction_volume):
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
        transaction_volume = get_tansaction_traffic_per_node(G, time, node, transaction_volume)
        
        betweenness_dict[(time, node)] = betweenness_nodes[node]
        closeness_dict[(time, node)] = closeness_nodes[node]


        for neighbor in neighbors:
            for k in range(len(G.get_edge_data(node, neighbor))):
                edge_amount = G.get_edge_data(node, neighbor)[k]['weight']
                initial_sent_amount = filtered_by_amount_per_node[tuple([time, connected_nodes[0]])]
                normalized_transactions.append([time, node, neighbor, edge_amount, edge_amount / initial_sent_amount])

    return normalized_transactions, transaction_volume

def plot_graph(G, title):
    """
    Visualise un graphe de transactions.

    Paramètres :
    - G (networkx.MultiDiGraph) : Graphe dirigé multi-arêtes.
    - title (str) : Titre du graphe.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Position des nœuds
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

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

def get_receivers(sender_id, all_day_transactions):
    """
    Obtient la liste des destinataires pour un expéditeur donné.

    Paramètres :
    - sender_id (any) : Identifiant de l'expéditeur.
    - all_day_transactions (list) : Liste des transactions d'une journée.

    Retourne :
    - list : Liste des destinataires.
    """
    receivers = []
    for transaction in all_day_transactions:
        if transaction[1] != sender_id:
            receivers.append(transaction[2])
    return receivers

def get_multiple_receivers(receivers):
    """
    Identifie les destinataires multiples dans une liste de destinataires.

    Paramètres :
    - receivers (list) : Liste des destinataires.

    Retourne :
    - set : Ensemble des destinataires multiples.
    """
    seen, multiple_receiver = [], set()
    for r in receivers:
        if r not in seen:
            seen.append(r)
        else:
            multiple_receiver.add(r)
    return multiple_receiver

def fix_multiple_receiver_amount(all_day_transactions, multiple_receivers):
    """
    Corrige les montants des transactions pour les destinataires multiples.

    Paramètres :
    - all_day_transactions (list) : Liste des transactions d'une journée.
    - multiple_receivers (set) : Ensemble des destinataires multiples.

    Retourne :
    - list : Liste des transactions corrigées.
    """
    for r in multiple_receivers:
        r_sum, edge_sum = 0, 0
        for transaction in all_day_transactions:
            if transaction[2] == r:
                r_sum += transaction[4]
                edge_sum += transaction[3]

        if commission_cut_off <= r_sum <= 1.:
            all_day_transactions.append([all_day_transactions[0][0], 'sender', r, edge_sum, r_sum])
    return all_day_transactions

def get_suspicious_transactions(all_normalized_transactions, commission_cut_off):
    """
    Identifie les transactions suspectes en fonction d'un seuil de commission.

    Paramètres :
    - all_normalized_transactions (list) : Liste des transactions normalisées.
    - commission_cut_off (float) : Seuil de commission pour détecter les transactions suspectes.

    Retourne :
    - list : Liste des transactions suspectes.
    """
    suspicious_transactions = []
    for all_day_transactions in all_normalized_transactions:
        sender_id = all_day_transactions[0][1]
        receivers = get_receivers(sender_id, all_day_transactions)
        multiple_receivers = get_multiple_receivers(receivers)
        if multiple_receivers:
            all_day_transactions = fix_multiple_receiver_amount(all_day_transactions, multiple_receivers)

        for transaction in all_day_transactions:
            if sender_id != transaction[1] and commission_cut_off <= transaction[4] <= 1.:
                suspicious_transactions.append(all_day_transactions)
    return suspicious_transactions

# Correction du problème des destinataires multiples
suspicious_transactions = get_suspicious_transactions(all_normalized_transactions, commission_cut_off)

def get_suspicious_traffic(transaction_volume):
    """
    Normalise le paramètre de trafic par le volume maximal de transactions.

    Paramètres :
    - transaction_volume (dict) : Dictionnaire du volume de transactions.

    Retourne :
    - dict : Dictionnaire du trafic suspect normalisé.
    """
    max_volume = max(transaction_volume.values())
    suspicious_traffic = {k: v / max_volume for k, v in transaction_volume.items()}
    return suspicious_traffic

# Normalisation du paramètre de trafic des transactions
suspicious_traffic = get_suspicious_traffic(transaction_volume)

def get_gamma(suspicious_traffic, betweenness_dict, closeness_dict):
    """
    Calcule le paramètre gamma pour chaque nœud, combinaison linéaire de trois caractéristiques du réseau.

    Paramètres :
    - suspicious_traffic (dict) : Dictionnaire du trafic suspect normalisé.
    - betweenness_dict (dict) : Dictionnaire de la centralité d'intermédiarité.
    - closeness_dict (dict) : Dictionnaire de la centralité de proximité.

    Retourne :
    - tuple : Liste des nœuds triés par gamma et dictionnaire des gamma.
    """
    gamma_day_node, gamma_dict = {}, {}
    a, b, c = 0.5, 0.4, 0.1  # Coefficients pour la combinaison linéaire
    for key in suspicious_traffic.keys():
        gamma_day_node[key] = a * suspicious_traffic[key] + b * betweenness_dict[key] + c * closeness_dict[key]

    for key in gamma_day_node.keys():
        if key[1] not in gamma_dict:
            gamma_dict[key[1]] = gamma_day_node[key]
        else:
            gamma_dict[key[1]] += gamma_day_node[key]

    sorted_gamma_nodes = sorted(gamma_dict.items(), key=lambda kv: -kv[1])
    sorted_gamma_nodes = [node[0] for node in sorted_gamma_nodes]
    return sorted_gamma_nodes, gamma_dict

gamma_nodes, gamma_dict = get_gamma(suspicious_traffic, betweenness_dict, closeness_dict)

def sort_suspicious_transactions(suspicious_transactions, gamma):
    """
    Classe les transactions suspectes en fonction du paramètre gamma des nœuds impliqués.

    Paramètres :
    - suspicious_transactions (list) : Liste des transactions suspectes.
    - gamma (dict) : Dictionnaire des paramètres gamma.

    Retourne :
    - list : Liste des transactions suspectes triées par ordre décroissant de suspicion.
    """
    suspicious_transactions_parameter = []
    for transaction_chain in suspicious_transactions:
        gamma_sum = 0
        for transaction in transaction_chain:
            if transaction[1] != 'sender':
                gamma_sum += gamma[transaction[1]] + gamma[transaction[2]]
        avg_gamma = gamma_sum / (2 * (len(transaction_chain)))
        if (gamma_sum, transaction_chain) not in suspicious_transactions_parameter:
            suspicious_transactions_parameter.append((avg_gamma, transaction_chain))

    suspicious_transactions_parameter = sorted(suspicious_transactions_parameter, key=lambda x: (-x[0], x[1]))
    rank_suspicious_transactions = [line[1] for line in suspicious_transactions_parameter]
    suspicious_transactions_by_line = [transaction for sublist in rank_suspicious_transactions for transaction in sublist]
    return suspicious_transactions_by_line

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

# Visualisation des graphes des 10 transactions les plus suspectes
top_10_suspicious_transactions = transaction_chains[:10]
for i, transaction_chain in enumerate(top_10_suspicious_transactions):
    edges = [(trans[1], trans[2], trans[3]) for trans in transaction_chain]
    G = create_graph(edges)
    plot_graph(G, f"Top {i+1} des transactions suspectes")
