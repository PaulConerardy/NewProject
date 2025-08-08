# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialisation de la session Spark
spark = SparkSession.builder \
    .appName("DecisionTree_AML_Rules_Calibration") \
    .getOrCreate()

# 2. Simulation des données d'entrée
# REMPLACEZ CETTE PARTIE PAR LE CHARGEMENT DE VOS DONNÉES
# Supposons que vos données ressemblent à ceci :
# id_alerte, montant, pays_destination, type_client, alerte_pertinente (1 ou 0)
data = [
    (1, 15000, "FRA", "Particulier", 0),
    (2, 8500, "USA", "Entreprise", 0),
    (3, 150000, "CYM", "Entreprise", 1), # Montant élevé vers un pays à risque
    (4, 2500, "DEU", "Particulier", 0),
    (5, 75000, "PAN", "Particulier", 1), # Montant élevé vers un pays à risque
    (6, 120000, "CHE", "Entreprise", 1),
    (7, 9500, "FRA", "Entreprise", 0),
    (8, 201000, "PAN", "Entreprise", 1),
    (9, 500, "ESP", "Particulier", 0),
    (10, 45000, "CYM", "Particulier", 0) # Montant élevé mais peut-être pas assez
]
columns = ["id_alerte", "montant", "pays_destination", "type_client", "alerte_pertinente"]
df = spark.createDataFrame(data, columns)

print("Aperçu des données brutes :")
df.show()

# 3. Préparation des données (Feature Engineering)

# a. Identifier les colonnes catégorielles et numériques
categorical_cols = ["pays_destination", "type_client"]
numerical_cols = ["montant"]
label_col = "alerte_pertinente"

# b. Créer les étapes de transformation pour les colonnes catégorielles
# StringIndexer convertit les chaînes de caractères en indices numériques.
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="keep") for col in categorical_cols]

# c. Assembler toutes les features dans un seul vecteur
# Les modèles ML de Spark ont besoin d'une colonne contenant un vecteur de toutes les features.
feature_cols = numerical_cols + [col + "_indexed" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# d. Définir le modèle
# On instancie l'arbre de décision
dt = DecisionTreeClassifier(labelCol=label_col, featuresCol="features", maxDepth=3, seed=42)

# e. Créer le Pipeline
# Un pipeline enchaîne toutes les étapes : indexation, assemblage, et modèle.
pipeline = Pipeline(stages=indexers + [assembler, dt])

# 4. Entraînement du modèle

# Diviser les données en un ensemble d'entraînement et de test
(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

print("\nEntraînement du modèle sur les données d'entraînement...")
model = pipeline.fit(training_data)
print("Entraînement terminé.")

# 5. Évaluation du modèle (Optionnel mais recommandé)
predictions = model.transform(test_data)

print("\nPrédictions sur l'ensemble de test :")
predictions.select("id_alerte", "montant", "pays_destination", "alerte_pertinente", "prediction", "probability").show()

evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"\nPrécision du modèle sur les données de test : {accuracy:.2f}")


# 6. Interprétation de l'arbre de décision pour extraire les règles

# C'est la partie la plus importante pour vous.
# On récupère le modèle d'arbre de décision à partir du pipeline entraîné.
tree_model = model.stages[-1]

# toDebugString nous donne la structure de l'arbre sous forme de texte.
tree_rules_string = tree_model.toDebugString

print("\n" + "="*50)
print("         RÈGLES DE DÉCISION EXTRAITES DE L'ARBRE")
print("="*50)
print(tree_rules_string)

# 7. Traduction des règles en langage humain

print("\n" + "="*50)
print("         TRADUCTION DES RÈGLES EN LANGAGE HUMAIN")
print("="*50)
print("Comment lire l'arbre ci-dessus :")
print("- 'feature 0' correspond à 'montant'")
print("- 'feature 1' correspond à 'pays_destination_indexed'")
print("- 'feature 2' correspond à 'type_client_indexed'")
print("Les valeurs pour les features indexées peuvent être retrouvées avec les modèles d'indexation.")
print("\nExemple de traduction d'une branche de l'arbre :")
print("IF (montant > 100000) AND (pays_destination IN ['PAN', 'CYM']) THEN Predire: Alerte Pertinente (1.0)")
print("IF (montant <= 50000) THEN Predire: Alerte Non Pertinente (0.0)")

# Pour obtenir les correspondances exactes entre les index et les catégories :
# (Utile pour une interprétation précise)
print("\nCorrespondances des index pour 'pays_destination':")
country_indexer_model = model.stages[0] # Le premier StringIndexer
country_labels = country_indexer_model.labels
print({i: label for i, label in enumerate(country_labels)})


# Arrêt de la session Spark
spark.stop()

