import networkx as nx
import pandas as pd
from config_loader import load_config, load_data

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

def fix_multiple_receiver_amount(all_day_transactions, multiple_receivers, commission_cut_off):
    """
    Corrige les montants des transactions pour les destinataires multiples.

    Paramètres :
    - all_day_transactions (list) : Liste des transactions d'une journée.
    - multiple_receivers (set) : Ensemble des destinataires multiples.
    - commission_cut_off (float) : Seuil de commission pour détecter les transactions suspectes.

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

def filter_transactions(df, money_cut_off):
    """
    Filtre les transactions en fonction d'un seuil de montant.

    Paramètres :
    - df (pd.DataFrame) : DataFrame contenant les données de transactions.
    - money_cut_off (float) : Seuil de montant pour filtrer les transactions.

    Retourne :
    - pd.DataFrame : DataFrame contenant les transactions filtrées.
    - dict : Dictionnaire des montants totaux par expéditeur et par jour.
    """
    agg = df.groupby(['date', 'SENDER'])['AMOUNT'].sum()
    agg = agg[agg > money_cut_off]  # Filtrage des transactions supérieures au seuil
    agg_flat = agg.reset_index()
    df_after_cutoff = pd.merge(agg_flat[['date', 'SENDER']], df, how='left')
    filtered_by_amount_per_node = agg.to_dict()
    return df_after_cutoff, filtered_by_amount_per_node

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
            all_day_transactions = fix_multiple_receiver_amount(all_day_transactions, multiple_receivers, commission_cut_off)

        for transaction in all_day_transactions:
            if sender_id != transaction[1] and commission_cut_off <= transaction[4] <= 1.:
                suspicious_transactions.append(all_day_transactions)
    return suspicious_transactions


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

def get_gamma(suspicious_traffic, betweenness_dict, closeness_dict, suspicious_transactions):
    """
    Calcule le paramètre gamma pour chaque nœud, combinaison linéaire de trois caractéristiques du réseau.

    Paramètres :
    - suspicious_traffic (dict) : Dictionnaire du trafic suspect normalisé.
    - betweenness_dict (dict) : Dictionnaire de la centralité d'intermédiarité.
    - closeness_dict (dict) : Dictionnaire de la centralité de proximité.
    - suspicious_transactions (list) : Liste des transactions suspectes.

    Retourne :
    - tuple : Liste des nœuds triés par gamma et dictionnaire des gamma.
    """
    gamma_day_node, gamma_dict = {}, {}
    a, b, c = 0.5, 0.4, 0.1  # Coefficients pour la combinaison linéaire

    for key in suspicious_traffic.keys():
        if key in betweenness_dict and key in closeness_dict:
            gamma_day_node[key] = (a * suspicious_traffic[key] +
                                    b * betweenness_dict[key] +
                                    c * closeness_dict[key])

    for key in gamma_day_node.keys():
        node = key[1]
        if node not in gamma_dict:
            gamma_dict[node] = gamma_day_node[key]
        else:
            gamma_dict[node] += gamma_day_node[key]

    # Ajouter les nœuds manquants avec une valeur gamma par défaut (par exemple, 0)
    for transaction_chain in suspicious_transactions:
        for transaction in transaction_chain:
            if transaction[1] != 'sender' and transaction[1] not in gamma_dict:
                gamma_dict[transaction[1]] = 0
            if transaction[2] not in gamma_dict:
                gamma_dict[transaction[2]] = 0

    sorted_gamma_nodes = sorted(gamma_dict.items(), key=lambda kv: -kv[1])
    sorted_gamma_nodes = [node[0] for node in sorted_gamma_nodes]
    return sorted_gamma_nodes, gamma_dict



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
        count = 0
        for transaction in transaction_chain:
            if transaction[1] != 'sender' and transaction[1] in gamma and transaction[2] in gamma:
                gamma_sum += gamma[transaction[1]] + gamma[transaction[2]]
                count += 2

        if count > 0:
            avg_gamma = gamma_sum / count
            if (gamma_sum, transaction_chain) not in suspicious_transactions_parameter:
                suspicious_transactions_parameter.append((avg_gamma, transaction_chain))

    suspicious_transactions_parameter = sorted(suspicious_transactions_parameter, key=lambda x: (-x[0], x[1]))
    rank_suspicious_transactions = [line[1] for line in suspicious_transactions_parameter]
    suspicious_transactions_by_line = [transaction for sublist in rank_suspicious_transactions for transaction in sublist]
    return suspicious_transactions_by_line

