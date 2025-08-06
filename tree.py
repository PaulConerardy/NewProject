import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulation et Chargement des Données ---
# REMPLACEZ CETTE PARTIE PAR LE CHARGEMENT DE VOS DONNÉES
# Vos données doivent contenir les caractéristiques des alertes (les paramètres de vos règles)
# et une colonne cible indiquant la décision de l'analyste (ex: 1 pour "Suspicion Confirmée", 0 pour "Faux Positif").

def generate_mock_data(num_samples=2000):
    """Génère un jeu de données simulé pour la démonstration."""
    data = {
        'montant_transaction': np.random.lognormal(mean=8, sigma=1.5, size=num_samples),
        'solde_compte_avant': np.random.lognormal(mean=9, sigma=2, size=num_samples),
        'nombre_transactions_24h': np.random.randint(1, 50, size=num_samples),
        'pays_destination_risque': np.random.choice(['Faible', 'Moyen', 'Élevé'], size=num_samples, p=[0.7, 0.2, 0.1]),
        'nouveau_beneficiaire': np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]),
        'type_transaction': np.random.choice(['Virement', 'Carte', 'Prélèvement'], size=num_samples, p=[0.5, 0.3, 0.2])
    }
    df = pd.DataFrame(data)

    # Création de la cible (décision de l'analyste) de manière semi-réaliste
    # Les transactions suspectes auront tendance à avoir des montants élevés, vers des pays à risque, etc.
    conditions = (
        (df['montant_transaction'] > 10000) & (df['pays_destination_risque'] == 'Élevé') |
        (df['nombre_transactions_24h'] > 30) |
        ((df['montant_transaction'] > 20000) & (df['nouveau_beneficiaire'] == 1))
    )
    df['decision_analyste'] = np.where(conditions, 1, 0)
    # Ajout d'un peu de bruit pour simuler l'incertitude humaine
    noise = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
    df['decision_analyste'] = df['decision_analyste'] ^ noise
    
    print("Aperçu des données générées :")
    print(df.head())
    print("\nDistribution de la décision de l'analyste :")
    print(df['decision_analyste'].value_counts(normalize=True))
    
    return df

# Charger les données
# Pour votre cas : df = pd.read_csv('votre_fichier_alertes.csv')
df = generate_mock_data()

# --- 2. Prétraitement des Données ---

# Définir la variable cible (Y) et les features (X)
X = df.drop('decision_analyste', axis=1)
y = df['decision_analyste']

# Identifier les colonnes catégorielles et numériques
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['number']).columns

print(f"\nFeatures catégorielles: {list(categorical_features)}")
print(f"Features numériques: {list(numerical_features)}")

# Créer un pipeline de prétraitement
# - OneHotEncoder pour les variables catégorielles (les transforme en 0 et 1)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 3. Division des données en ensembles d'entraînement et de test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 4. Entraînement et Optimisation du Modèle ---

# Définir le modèle
# class_weight='balanced' est important si les cas de suspicion sont rares
dt_classifier = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Créer le pipeline complet incluant le prétraitement et le modèle
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', dt_classifier)])

# Définir la grille de paramètres à tester pour la calibration (Grid Search)
# C'est ici que l'on calibre le modèle pour trouver les "meilleures" règles
param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 4, 5, 6, 8], # Profondeur de l'arbre (complexité des règles)
    'classifier__min_samples_leaf': [10, 20, 30, 50], # Nb min d'alertes dans une feuille finale
    'classifier__min_samples_split': [20, 40, 60, 100] # Nb min d'alertes pour créer une nouvelle règle
}

# Recherche par grille avec validation croisée pour trouver les meilleurs paramètres
# cv=5 signifie que l'on teste chaque combinaison 5 fois sur des sous-ensembles différents des données
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
print("\nLancement de l'optimisation des hyperparamètres (GridSearchCV)...")
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres trouvés
print(f"\nMeilleurs paramètres trouvés : {grid_search.best_params_}")

# Le meilleur modèle est directement accessible
best_model = grid_search.best_estimator_

# --- 5. Évaluation du Modèle ---

print("\n--- Évaluation du Modèle sur l'ensemble de Test ---")
y_pred = best_model.predict(X_test)

print("\nMatrice de Confusion :")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Faux Positif', 'Suspicion'], yticklabels=['Faux Positif', 'Suspicion'])
plt.ylabel('Vraie étiquette')
plt.xlabel('Étiquette prédite')
plt.title('Matrice de Confusion')
plt.show()


print("\nRapport de Classification :")
# Precision: Sur tout ce que le modèle a prédit comme "Suspicion", quelle part l'était vraiment ?
# Recall: Sur toutes les "Suspicion" réelles, combien le modèle en a-t-il trouvé ? (Très important en LCB)
# F1-score: Moyenne harmonique de Precision et Recall
print(classification_report(y_test, y_pred, target_names=['Faux Positif', 'Suspicion']))


# --- 6. Interprétation des Résultats pour la Calibration ---

# a) Visualisation de l'arbre de décision
# C'est la partie la plus importante pour comprendre les règles apprises par le modèle.
print("\n--- Visualisation de l'Arbre de Décision ---")
# Récupérer les noms des features après One-Hot Encoding
feature_names = list(numerical_features) + \
                list(best_model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features))

plt.figure(figsize=(25, 15))
plot_tree(best_model.named_steps['classifier'],
          feature_names=feature_names,
          class_names=['Faux Positif', 'Suspicion'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=4) # Limiter la profondeur pour la lisibilité
plt.title("Arbre de Décision Modélisant les Décisions des Analystes (Profondeur=4)", fontsize=20)
plt.show()

# b) Quantification de l'importance des paramètres
print("\n--- Importance des Paramètres (Feature Importance) ---")
# Cela vous indique quels paramètres ont le plus d'influence sur la décision finale.
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10)) # Top 10
plt.title('Top 10 des Paramètres les plus Influents')
plt.xlabel('Importance (Gini Importance)')
plt.ylabel('Paramètre')
plt.tight_layout()
plt.show()

print("\nTableau de l'importance des paramètres :")
print(feature_importance_df)
