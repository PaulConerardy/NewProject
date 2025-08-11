# Import des librairies nécessaires
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ==============================================================================
# 1. Initialisation de la SparkSession
# ==============================================================================
spark = SparkSession.builder \
    .appName("AMLRulesCalibration") \
    .getOrCreate()

print(f"Version de Spark : {spark.version}")

# ==============================================================================
# 2. Création d'un jeu de données d'exemple (simulant vos alertes)
# ==============================================================================
# Chaque ligne représente une alerte qui a été générée et analysée.
# is_confirmed_suspicious: 1 = Vrai Positif, 0 = Faux Positif. C'est notre variable cible.
data = [
    (1, 101, 15000.0, 'FRA', 'CHE', True, 5, 1),   # Très suspect
    (0, 102, 800.0, 'FRA', 'FRA', False, 2, 0),    # Normal
    (0, 103, 9500.0, 'ESP', 'FRA', False, 1, 0),   # FP - Montant élevé mais contexte normal
    (1, 104, 12000.0, 'BEL', 'CYP', True, 1, 1),   # Suspect - Pays et nouveau bénéficiaire
    (0, 101, 500.0, 'FRA', 'DEU', False, 12, 0),   # FP - Client connu, petit montant
    (1, 105, 50000.0, 'LUX', 'PAN', True, 1, 1),   # Très suspect - Montant et destination
    (0, 106, 25000.0, 'FRA', 'USA', False, 3, 0),  # FP - Montant élevé mais contexte OK
    (1, 107, 2000.0, 'NLD', 'TUR', True, 8, 1),    # Suspect - Nouveau bénéficiaire et vélocité
    (1, 101, 9800.0, 'FRA', 'LVA', True, 4, 1),    # Suspect - Destination sensible
    (0, 108, 11000.0, 'FRA', 'GBR', False, 2, 0)   # FP - Juste au-dessus du seuil actuel
]
columns = ["is_confirmed_suspicious", "customer_id", "transaction_amount", "country_origin", 
           "country_destination", "is_new_beneficiary", "tx_count_24h", "label"]

df = spark.createDataFrame(data, columns)

print("Aperçu des données brutes :")
df.show()

# ==============================================================================
# 3. Prétraitement et Ingénierie des Caractéristiques (Feature Engineering)
# ==============================================================================

# Étape 1: Indexer les colonnes catégorielles (string -> index numérique)
# On transforme les pays en chiffres pour que le modèle puisse les utiliser.
country_origin_indexer = StringIndexer(inputCol="country_origin", outputCol="country_origin_idx")
country_dest_indexer = StringIndexer(inputCol="country_destination", outputCol="country_dest_idx")

# Étape 2: Assembler toutes les caractéristiques dans un seul vecteur
# L'arbre de décision prend un seul vecteur de features en entrée.
# Note : on inclut les colonnes numériques et les nouvelles colonnes indexées.
# On exclut les identifiants, la variable cible 'label' et les colonnes brutes.
feature_cols = [
    'transaction_amount', 
    'country_origin_idx', 
    'country_dest_idx',
    'is_new_beneficiary', # Est déjà numérique (True/False -> 1.0/0.0)
    'tx_count_24h'
]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ==============================================================================
# 4. Création du Pipeline de Machine Learning
# ==============================================================================

# Définition du modèle : Arbre de Décision
# labelCol : la colonne qui contient la vérité (0 ou 1)
# featuresCol : la colonne qui contient le vecteur de caractéristiques
# maxDepth : profondeur max de l'arbre, à ajuster pour éviter le sur-apprentissage
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=4)

# Création du Pipeline qui enchaîne toutes les étapes
pipeline = Pipeline(stages=[
    country_origin_indexer, 
    country_dest_indexer, 
    vector_assembler, 
    dt
])

# ==============================================================================
# 5. Entraînement et Prédiction
# ==============================================================================

# Division des données : 80% pour l'entraînement, 20% pour le test
(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

print(f"Nombre d'alertes pour l'entraînement : {training_data.count()}")
print(f"Nombre d'alertes pour le test : {test_data.count()}")

# Entraînement du modèle sur les données d'entraînement
model = pipeline.fit(training_data)

# Application du modèle sur les données de test pour faire des prédictions
predictions = model.transform(test_data)

print("Aperçu des prédictions sur le jeu de test :")
predictions.select("features", "label", "prediction", "probability").show(truncate=False)

# ==============================================================================
# 6. Évaluation du Modèle
# ==============================================================================

# Utilisation d'un évaluateur pour mesurer la performance
# 'f1' est une bonne métrique pour des données déséquilibrées (fréquent en AML)
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

print(f"Accuracy du modèle sur les données de test = {accuracy:.2%}")
print(f"F1-Score du modèle sur les données de test = {f1_score:.2%}")

# ==============================================================================
# 7. Interprétation de l'Arbre et Extraction des Règles
# ==============================================================================

# On extrait le modèle d'arbre de décision entraîné depuis le pipeline
tree_model = model.stages[-1]

# toDebugString nous donne la structure complète de l'arbre
print("\n--- Structure de l'Arbre de Décision (Règles Détectées) ---\n")
print(tree_model.toDebugString)

print("\n--- Comment Interpréter l'Arbre Ci-Dessus ---\n")
print("L'arbre se lit de haut en bas. Chaque 'if' est une condition sur une feature.")
print("  - 'feature 0' correspond à la 1ère colonne dans 'feature_cols' -> 'transaction_amount'")
print("  - 'feature 1' correspond à la 2ème -> 'country_origin_idx'")
print("  - 'feature 2' correspond à la 3ème -> 'country_dest_idx'")
print("  - 'feature 3' correspond à la 4ème -> 'is_new_beneficiary'")
print("  - 'feature 4' correspond à la 5ème -> 'tx_count_24h'")
print("\n  - 'Predict: 1.0' signifie que le modèle prédit une alerte suspecte (Vrai Positif).")
print("  - 'Predict: 0.0' signifie que le modèle prédit un Faux Positif.")
print("\nExemple de traduction d'un chemin en règle métier :")
print("SI 'feature 0' (transaction_amount) <= 9650.0 ET 'feature 2' (country_dest_idx) <= 1.5 ... ALORS Predict: 1.0")
print("Cela peut se traduire par : 'SI Montant <= 9650€ ET Pays Destination est (par ex.) LVA ou CHE, ALORS générer une alerte'.")

# Pour connaître la correspondance entre index et pays
print("\n--- Correspondance Index -> Valeur pour les variables catégorielles ---")

# Fonction pour afficher la correspondance
def display_indexer_mappings(model, stage_index, col_name):
    indexer_model = model.stages[stage_index]
    labels = indexer_model.labels
    print(f"Mapping pour la colonne '{col_name}':")
    for i, label in enumerate(labels):
        print(f"  Index {i} -> {label}")

display_indexer_mappings(model, 0, "country_origin")
display_indexer_mappings(model, 1, "country_destination")

spark.stop()
