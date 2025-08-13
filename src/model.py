import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Étape 1: Chargement des Données ---
# Adaptez les noms de fichiers et les séparateurs si nécessaire (ex: sep=';')
print("Chargement des fichiers...")
try:
    alerts_df = pd.read_csv('alerts.csv')
    transactions_df = pd.read_csv('transactions.csv')
    customers_df = pd.read_csv('customers.csv')
except FileNotFoundError as e:
    print(f"Erreur: {e}. Assurez-vous que les fichiers CSV sont dans le même répertoire que le script.")
    exit()

print("Fichiers chargés avec succès.")

# --- Étape 2: Pré-traitement et Ingénierie des Caractéristiques ---
print("Début du pré-traitement...")

# Conversion des colonnes de date en format datetime (ADAPTER LES NOMS DE COLONNES)
# Supposons que la colonne de date s'appelle 'alert_date' dans alerts_df
# et 'transaction_date' dans transactions_df.
alerts_df['alert_date'] = pd.to_datetime(alerts_df['alert_date'])
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

# Création d'un identifiant de semaine ('week_id') pour le matching
# Format: Année-NuméroDeSemaine (ex: '2025-33')
alerts_df['week_id'] = alerts_df['alert_date'].dt.to_period('W').astype(str)
transactions_df['week_id'] = transactions_df['transaction_date'].dt.to_period('W').astype(str)

# Agrégation des données transactionnelles par client et par semaine
# C'est ici que vous créez les variables que l'arbre de décision va analyser
print("Agrégation des données transactionnelles...")
transaction_agg_df = transactions_df.groupby(['customer_id', 'week_id']).agg(
    # Métriques sur les montants
    montant_total_hebdo=('amount', 'sum'),
    montant_moyen_hebdo=('amount', 'mean'),
    montant_max_hebdo=('amount', 'max'),
    ecart_type_montant_hebdo=('amount', 'std'),
    
    # Métriques sur le nombre de transactions
    nb_transactions_hebdo=('amount', 'count'),
    
    # Exemple de métrique plus complexe: nombre de transactions vers des pays à risque
    # Vous devez avoir une colonne 'counterparty_country' et une liste de pays à risque
    # nb_trans_pays_risque=('counterparty_country', lambda x: x.isin(['PAYS_A', 'PAYS_B']).sum())
).reset_index()

# Remplacer les NaN dans l'écart-type (pour les clients avec 1 seule transaction) par 0
transaction_agg_df['ecart_type_montant_hebdo'] = transaction_agg_df['ecart_type_montant_hebdo'].fillna(0)


# --- Étape 3: Fusion des Données ---
print("Fusion des jeux de données...")

# Fusionner les transactions agrégées avec les données clients
# Supposons que l'ID client s'appelle 'customer_id' partout
merged_df = pd.merge(transaction_agg_df, customers_df, on='customer_id', how='left')

# Fusionner le résultat avec les données d'alertes
# Nous utilisons une fusion 'droite' (right merge) pour ne garder que les semaines/clients qui ont généré une alerte
# C'est le coeur de l'analyse : on veut comprendre ce qui caractérise les alertes
final_df = pd.merge(merged_df, alerts_df, on=['customer_id', 'week_id'], how='right')

# --- Étape 4: Préparation pour le Modèle ---
print("Préparation des données pour le modèle...")

# !! POINT CRUCIAL : Définir la variable cible (target) !!
# Vous devez avoir une colonne dans 'alerts.csv' qui indique si l'alerte était pertinente
# Par exemple, si elle a mené à une Déclaration de Soupçon (DS)
# Ici, nous supposons une colonne 'decision' avec les valeurs 'DS' (positif) et 'Fausse_Alerte' (négatif)
# C'est la variable que nous voulons prédire.
if 'decision' not in final_df.columns:
    print("ERREUR: La colonne 'decision' (notre cible) est manquante dans le fichier d'alertes.")
    print("Vous devez disposer d'un historique sur la pertinence des alertes passées.")
    exit()

# Transformation de la cible en variable binaire (0 ou 1)
final_df['target'] = final_df['decision'].apply(lambda x: 1 if x == 'DS' else 0)

# Sélection des caractéristiques (features) pour le modèle
# On exclut les identifiants, dates, et la cible textuelle originale
features = [col for col in final_df.columns if col not in [
    'customer_id', 'week_id', 'alert_date', 'decision', 'target', 'alert_id' # Adaptez cette liste
]]

X = final_df[features]
y = final_df['target']

# Encodage des variables catégorielles (ex: secteur d'activité du client)
# 'get_dummies' transforme les catégories en colonnes binaires (0/1)
X = pd.get_dummies(X, drop_first=True)

# Gérer les valeurs manquantes restantes (stratégie simple : remplacer par 0)
X = X.fillna(0)

# Division en ensembles d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Taille du jeu d'entraînement: {X_train.shape[0]} échantillons")
print(f"Taille du jeu de test: {X_test.shape[0]} échantillons")

# --- Étape 5: Entraînement de l'Arbre de Décision ---
print("Entraînement de l'arbre de décision...")

# On limite la profondeur de l'arbre (max_depth) pour qu'il reste simple et interprétable.
# C'est la clé pour éviter le sur-apprentissage et obtenir des règles claires.
# Commencez avec une profondeur de 3 ou 4.
decision_tree = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=20)
decision_tree.fit(X_train, y_train)

print("Modèle entraîné.")

# --- Étape 6: Évaluation et Visualisation ---
print("\n--- Évaluation du Modèle ---")
y_pred = decision_tree.predict(X_test)

# Matrice de confusion: montre les vrais positifs, faux positifs, etc.
print("Matrice de Confusion :")
print(confusion_matrix(y_test, y_pred))

# Rapport de classification: Précision, Rappel, F1-score
# - Précision: Sur toutes les alertes prédites comme 'DS', combien l'étaient vraiment ? (Limite les faux positifs)
# - Rappel (Recall): Sur toutes les vraies 'DS', combien ont été détectées ? (Limite les faux négatifs)
print("\nRapport de Classification :")
print(classification_report(y_test, y_pred, target_names=['Fausse_Alerte', 'DS']))

# Visualisation de l'arbre de décision
print("\nGénération de la visualisation de l'arbre... (peut prendre un moment)")
plt.figure(figsize=(25, 15))
plot_tree(decision_tree,
          feature_names=X.columns,
          class_names=['Fausse_Alerte', 'DS'],
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Arbre de Décision pour la Calibration des Alertes AML", fontsize=20)
plt.savefig("arbre_decision_aml.png")
print("\nL'arbre de décision a été sauvegardé dans 'arbre_decision_aml.png'")
plt.show()

