Voici un script PySpark simple pour déterminer l'importance des variables dans un problème de classification binaire :

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import pandas as pd

# Initialiser Spark
spark = SparkSession.builder \
    .appName("FeatureImportance") \
    .getOrCreate()

# Exemple : créer un dataset synthétique (remplacez par votre propre dataset)
data = [(1.0, 2.0, 3.0, 5.0, 1),
        (2.0, 3.0, 4.0, 6.0, 0),
        (3.0, 4.0, 5.0, 7.0, 1),
        (4.0, 5.0, 6.0, 8.0, 0),
        (5.0, 6.0, 7.0, 9.0, 1)]

columns = ["feature1", "feature2", "feature3", "feature4", "label"]
df = spark.createDataFrame(data, columns)

# Ou charger votre dataset
# df = spark.read.csv("path/to/your/data.csv", header=True, inferSchema=True)

# Préparer les features
feature_columns = [col for col in df.columns if col != "label"]
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

# Diviser les données (train/test)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Entraîner le modèle
model = pipeline.fit(train_df)

# Faire des prédictions
predictions = model.transform(test_df)

# Évaluer le modèle
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.3f}")

# Extraire l'importance des variables
rf_model = model.stages[1]  # Le Random Forest est le 2ème stage
feature_importance = rf_model.featureImportances.toArray()

# Créer un DataFrame avec les importances
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nImportance des variables :")
print(importance_df)

# Visualisation simple (optionnel)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Importance des Variables - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Méthode alternative : utiliser d'autres algorithmes pour comparer
from pyspark.ml.classification import GBTClassifier

# Gradient Boosted Trees
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

pipeline_gbt = Pipeline(stages=[assembler, gbt])
model_gbt = pipeline_gbt.fit(train_df)
gbt_model = model_gbt.stages[1]

# Importance avec GBT
gbt_importance = gbt_model.featureImportances.toArray()
importance_gbt_df = pd.DataFrame({
    'feature': feature_columns,
    'importance_gbt': gbt_importance
}).sort_values('importance_gbt', ascending=False)

print("\nImportance des variables (GBT) :")
print(importance_gbt_df)

# Comparer les deux méthodes
comparison_df = importance_df.merge(importance_gbt_df, on='feature')
print("\nComparaison RF vs GBT :")
print(comparison_df)

# Fermer Spark
spark.stop()
```

Ce script fait les choses suivantes :

1. **Initialise une session Spark** et charge les données
2. **Prépare les features** avec VectorAssembler
3. **Utilise Random Forest** comme algorithme principal pour calculer l'importance
4. **Évalue le modèle** avec l'AUC
5. **Extrait et affiche l'importance** des variables triées par ordre décroissant
6. **Compare avec Gradient Boosted Trees** pour validation
7. **Inclut une visualisation** simple des importances

Pour utiliser ce script avec vos données :
- Remplacez la création du dataset synthétique par le chargement de vos données
- Ajustez les noms des colonnes selon votre dataset
- Modifiez les hyperparamètres des modèles si nécessaire

L'importance des variables vous aidera à identifier quelles features contribuent le plus à la classification binaire.