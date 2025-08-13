Voici le script modifié pour utiliser uniquement des opérations sur des DataFrames PySpark :

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Initialiser Spark
spark = SparkSession.builder \
    .appName("FeatureImportancePySpark") \
    .getOrCreate()

# Créer un dataset synthétique avec PySpark DataFrame
schema = StructType([
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("feature3", DoubleType(), True),
    StructField("feature4", DoubleType(), True),
    StructField("label", IntegerType(), True)
])

data = [(1.0, 2.0, 3.0, 5.0, 1),
        (2.0, 3.0, 4.0, 6.0, 0),
        (3.0, 4.0, 5.0, 7.0, 1),
        (4.0, 5.0, 6.0, 8.0, 0),
        (5.0, 6.0, 7.0, 9.0, 1),
        (6.0, 7.0, 8.0, 10.0, 0),
        (7.0, 8.0, 9.0, 11.0, 1),
        (8.0, 9.0, 10.0, 12.0, 0)]

df = spark.createDataFrame(data, schema)

# Ou charger vos données
# df = spark.read.csv("path/to/your/data.csv", header=True, inferSchema=True)

# Afficher les données
print("Dataset:")
df.show()

# Obtenir les noms des features (exclure la colonne label)
feature_columns = [col_name for col_name in df.columns if col_name != "label"]

# Préparer les features
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

# Créer le modèle Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    seed=42
)

# Créer le pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Diviser les données avec PySpark
train_df = df.sample(fraction=0.8, seed=42)
test_df = df.subtract(train_df)

print(f"Taille train: {train_df.count()}")
print(f"Taille test: {test_df.count()}")

# Entraîner le modèle
model = pipeline.fit(train_df)

# Faire des prédictions
predictions = model.transform(test_df)

# Afficher les prédictions
print("Prédictions:")
predictions.select("label", "prediction", "probability").show()

# Évaluer le modèle
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.3f}")

# Extraire l'importance des variables avec PySpark DataFrame
rf_model = model.stages[1]
feature_importance_array = rf_model.featureImportances.toArray()

# Créer un DataFrame PySpark avec les importances
importance_data = list(zip(feature_columns, feature_importance_array))
importance_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("importance", DoubleType(), True)
])

from pyspark.sql.types import StringType
importance_df = spark.createDataFrame(importance_data, importance_schema)

# Trier par importance décroissante
importance_df_sorted = importance_df.orderBy(desc("importance"))

print("\nImportance des variables (Random Forest):")
importance_df_sorted.show()

# Ajouter un rang pour chaque feature
window_spec = Window.orderBy(desc("importance"))
importance_with_rank = importance_df_sorted.withColumn(
    "rank", 
    row_number().over(window_spec)
)

print("Importance avec rang:")
importance_with_rank.show()

# Alternative avec Gradient Boosted Trees
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=50
)

pipeline_gbt = Pipeline(stages=[assembler, gbt])
model_gbt = pipeline_gbt.fit(train_df)
gbt_model = model_gbt.stages[1]

# Extraire importance GBT
gbt_importance_array = gbt_model.featureImportances.toArray()
gbt_importance_data = list(zip(feature_columns, gbt_importance_array))

gbt_importance_df = spark.createDataFrame(gbt_importance_data, importance_schema)
gbt_importance_df_sorted = gbt_importance_df.orderBy(desc("importance"))

print("\nImportance des variables (GBT):")
gbt_importance_df_sorted.show()

# Comparer les deux méthodes avec un join
comparison_df = importance_df.alias("rf").join(
    gbt_importance_df.alias("gbt"), 
    col("rf.feature") == col("gbt.feature")
).select(
    col("rf.feature").alias("feature"),
    col("rf.importance").alias("rf_importance"),
    col("gbt.importance").alias("gbt_importance")
).orderBy(desc("rf_importance"))

print("Comparaison RF vs GBT:")
comparison_df.show()

# Calculer des statistiques sur les importances
print("Statistiques des importances RF:")
importance_df.select("importance").describe().show()

# Filtrer les features les plus importantes (seuil > moyenne)
avg_importance = importance_df.agg({"importance": "avg"}).collect()[0][0]
important_features = importance_df.filter(col("importance") > avg_importance)

print(f"Features au-dessus de la moyenne ({avg_importance:.3f}):")
important_features.orderBy(desc("importance")).show()

# Calculer le pourcentage d'importance cumulée
total_importance = importance_df.agg({"importance": "sum"}).collect()[0][0]
importance_with_percentage = importance_df_sorted.withColumn(
    "importance_percentage", 
    (col("importance") / total_importance) * 100
)

print("Importance avec pourcentages:")
importance_with_percentage.show()

# Sauvegarder les résultats (optionnel)
# importance_df_sorted.write.mode("overwrite").csv("output/feature_importance")
# comparison_df.write.mode("overwrite").csv("output/comparison")

# Fermer Spark
spark.stop()
```

Les principales modifications apportées :

1. **Création de DataFrames PySpark** : Utilisation de `spark.createDataFrame()` avec des schémas explicites
2. **Opérations sur DataFrames** : Remplacement de pandas par des opérations PySpark natives
3. **Tri et filtrage** : Utilisation de `orderBy()`, `filter()`, et `select()`
4. **Fonctions d'agrégation** : `agg()`, `describe()`, `sum()`, `avg()`
5. **Fonctions Window** : `row_number()` pour ajouter des rangs
6. **Joins** : Comparaison des méthodes avec `join()`
7. **Calculs de pourcentages** : Avec des opérations de colonnes PySpark
8. **Pas de conversion vers pandas** : Tout reste dans l'écosystème PySpark

Ce script est maintenant entièrement basé sur des opérations de DataFrames PySpark et peut gérer de gros volumes de données de manière distribuée.