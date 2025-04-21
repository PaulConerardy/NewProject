import pandas as pd
import networkx as nx
from pyvis.network import Network
import re
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths to data files
data_dir = "data/"
clients_path = data_dir + "clients.csv"
transactions_path = data_dir + "transactions.csv"
sars_path = data_dir + "sars.csv"
alerts_path = data_dir + "alerts.csv"

def load_data():
    clients = pd.read_csv(clients_path)
    transactions = pd.read_csv(transactions_path)
    sars = pd.read_csv(sars_path)
    alerts = pd.read_csv(alerts_path)
    return clients, transactions, sars, alerts

def entity_resolution(name, clients_df):
    # Simple entity resolution: match full_name or business_name
    name = name.lower().strip()
    for _, row in clients_df.iterrows():
        if pd.notnull(row.get('full_name')) and name == str(row['full_name']).lower().strip():
            return row['client_id']
        if pd.notnull(row.get('business_name')) and name == str(row['business_name']).lower().strip():
            return row['client_id']
    return None

def extract_accomplices(narrative):
    # Extract names from narrative (very basic, can be improved)
    accomplices = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)", narrative)
    return accomplices

def build_knowledge_graph(client_id, clients, transactions, sars, alerts):
    G = nx.MultiDiGraph()
    # Add client node
    client_row = clients[clients['client_id'] == client_id]
    if client_row.empty:
        raise ValueError(f"Client ID {client_id} not found.")
    client_name = client_row.iloc[0].get('full_name') or client_row.iloc[0].get('business_name')
    G.add_node(client_id, label=client_name, type='client')
    # Add SARs and flagged transactions only
    client_sars = sars[sars['client_id'] == client_id].copy()
    client_sars['filing_date'] = pd.to_datetime(client_sars['filing_date'])
    client_sars = client_sars.sort_values('filing_date')
    flagged_trx_ids = set()
    sar_dates = {}
    for _, sar in client_sars.iterrows():
        sar_id = sar['sar_id']
        filing_date = sar['filing_date']
        sar_dates[sar_id] = filing_date
        G.add_node(sar_id, label=f"SAR {sar_id}", type='sar', date=filing_date.strftime('%Y-%m-%d'))
        G.add_edge(client_id, sar_id, label='subject')
        try:
            related_trx = ast.literal_eval(sar['related_transactions'])
        except Exception:
            related_trx = []
        for trx_id in related_trx:
            flagged_trx_ids.add(trx_id)
        accomplices = []
        if pd.notnull(sar.get('accomplices')) and sar['accomplices']:
            try:
                accomplices = ast.literal_eval(sar['accomplices'])
            except Exception:
                accomplices = []
        if not accomplices and pd.notnull(sar.get('narrative')):
            accomplices = extract_accomplices(sar['narrative'])
        for acc in accomplices:
            acc_id = entity_resolution(acc, clients)
            if acc_id:
                G.add_node(acc_id, label=acc, type='client')
                G.add_edge(sar_id, acc_id, label='accomplice')
            else:
                G.add_node(acc, label=acc, type='accomplice')
                G.add_edge(sar_id, acc, label='accomplice')
    # Add only flagged transactions and their edges
    client_trx = transactions[(transactions['client_id'] == client_id) & (transactions['transaction_id'].isin(flagged_trx_ids))].copy()
    client_trx['date'] = pd.to_datetime(client_trx['date'])
    client_trx = client_trx.sort_values('date')
    trx_dates = {}
    for _, trx in client_trx.iterrows():
        trx_id = trx['transaction_id']
        trx_date = trx['date']
        trx_dates[trx_id] = trx_date
        G.add_node(trx_id, label=f"TRX {trx_id}", type='transaction', date=trx_date.strftime('%Y-%m-%d'), amount=trx['amount'])
        G.add_edge(client_id, trx_id, label='initiated')
        cp_name = trx['counterparty_name']
        cp_id = entity_resolution(cp_name, clients)
        if cp_id:
            G.add_node(cp_id, label=cp_name, type='client')
            G.add_edge(trx_id, cp_id, label='to')
        else:
            G.add_node(cp_name, label=cp_name, type='counterparty')
            G.add_edge(trx_id, cp_name, label='to')
    # Link flagged transactions to SARs
    for _, sar in client_sars.iterrows():
        sar_id = sar['sar_id']
        try:
            related_trx = ast.literal_eval(sar['related_transactions'])
        except Exception:
            related_trx = []
        for trx_id in related_trx:
            if G.has_node(trx_id):
                G.add_edge(trx_id, sar_id, label='flagged', date=trx_dates.get(trx_id, ''))
    # Add alerts for this client
    client_alerts = alerts[alerts['client_id'] == client_id].copy()
    if not client_alerts.empty:
        client_alerts['alert_date'] = pd.to_datetime(client_alerts['alert_date'])
        for _, alert in client_alerts.iterrows():
            alert_id = alert['alert_id']
            alert_date = alert['alert_date']
            G.add_node(alert_id, label=f"Alert {alert_id}", type='alert', date=alert_date.strftime('%Y-%m-%d'), alert_type=alert['alert_type'], status=alert['status'])
            G.add_edge(client_id, alert_id, label='alerted')
            # Link alert to SAR if present
            if pd.notnull(alert.get('linked_sar_id')) and alert['linked_sar_id']:
                G.add_edge(alert_id, alert['linked_sar_id'], label='linked_sar')
            # Link alert to triggered transactions
            try:
                triggered_trx = ast.literal_eval(alert['triggered_by_transactions'])
            except Exception:
                triggered_trx = []
            for trx_id in triggered_trx:
                if G.has_node(trx_id):
                    G.add_edge(alert_id, trx_id, label='triggered')
    return G

def visualize_graph(G, output_html="client_kg.html"):
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    color_map = {'client': 'skyblue', 'transaction': 'orange', 'sar': 'red', 'counterparty': 'lightgreen', 'accomplice': 'pink', 'alert': 'purple'}
    nodes_with_dates = [(n, d) for n, d in G.nodes(data=True) if 'date' in d]
    nodes_without_dates = [(n, d) for n, d in G.nodes(data=True) if 'date' not in d]
    nodes_with_dates.sort(key=lambda x: x[1]['date'])
    for node, data in nodes_with_dates + nodes_without_dates:
        shape = 'ellipse'
        if data.get('type') == 'sar':
            shape = 'box'
        elif data.get('type') == 'transaction':
            shape = 'diamond'
        elif data.get('type') == 'alert':
            shape = 'star'
        net.add_node(node, label=data.get('label', str(node)), color=color_map.get(data.get('type'), 'gray'), title=str(data), shape=shape)
    for src, dst, data in G.edges(data=True):
        edge_label = data.get('label', '')
        if 'date' in data and data['date']:
            edge_label += f" ({data['date']})"
        net.add_edge(src, dst, label=edge_label)
    net.write_html(output_html)
    print(f"Graph saved to {output_html}")

def visualize_temporal_graph(G, output_png="client_kg_temporal.png"):
    sar_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == 'sar' and 'date' in d]
    trx_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == 'transaction' and 'date' in d]
    alert_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == 'alert' and 'date' in d]
    all_nodes = sar_nodes + trx_nodes + alert_nodes
    all_nodes.sort(key=lambda x: x[1]['date'])
    x_pos = {}
    for idx, (n, d) in enumerate(all_nodes):
        x_pos[n] = idx
    fig, ax = plt.subplots(figsize=(max(8, len(all_nodes)*2), 6))
    y_base = 0
    for n, d in all_nodes:
        if d['type'] == 'sar':
            color = 'red'
            shape = 's'
        elif d['type'] == 'transaction':
            color = 'orange'
            shape = 'D'
        elif d['type'] == 'alert':
            color = 'purple'
            shape = '*'
        else:
            color = 'gray'
            shape = 'o'
        ax.scatter(x_pos[n], y_base, c=color, marker=shape, s=400, zorder=3)
        ax.text(x_pos[n], y_base+0.2, d.get('label', str(n)), ha='center', fontsize=10, rotation=30)
        ax.text(x_pos[n], y_base-0.25, d.get('date',''), ha='center', fontsize=8, color='gray')
    client_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'client']
    if client_nodes:
        client_n = client_nodes[0]
        ax.scatter(-1, y_base, c='skyblue', marker='o', s=400, zorder=3)
        ax.text(-1, y_base+0.2, G.nodes[client_n].get('label', str(client_n)), ha='center', fontsize=10)
    for src, dst, data in G.edges(data=True):
        if src in x_pos and dst in x_pos:
            ax.annotate('', xy=(x_pos[dst], y_base), xytext=(x_pos[src], y_base),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1), zorder=2)
        elif src in x_pos and client_nodes and dst == client_nodes[0]:
            ax.annotate('', xy=(-1, y_base), xytext=(x_pos[src], y_base),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1), zorder=2)
        elif dst in x_pos and client_nodes and src == client_nodes[0]:
            ax.annotate('', xy=(x_pos[dst], y_base), xytext=(-1, y_base),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1), zorder=2)
    legend_handles = [mpatches.Patch(color='skyblue', label='Client'),
                      mpatches.Patch(color='orange', label='Transaction'),
                      mpatches.Patch(color='red', label='SAR'),
                      mpatches.Patch(color='purple', label='Alert')]
    ax.legend(handles=legend_handles, loc='upper left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Temporal Knowledge Graph (SARs, Transactions & Alerts)')
    ax.set_xlim(-2, len(all_nodes))
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Temporal graph saved to {output_png}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AML SAR Diagnostic Knowledge Graph Tool")
    parser.add_argument("client_id", type=str, help="Client ID to analyze")
    parser.add_argument("--output", type=str, default="client_kg.html", help="Output HTML file for graph visualization")
    args = parser.parse_args()
    clients, transactions, sars, alerts = load_data()
    G = build_knowledge_graph(args.client_id, clients, transactions, sars, alerts)
    visualize_graph(G, args.output)
    visualize_temporal_graph(G)

if __name__ == "__main__":
    main()