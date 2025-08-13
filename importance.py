Voici le script modifié pour éviter l'utilisation de VectorAssembler :

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, row_number, array, struct, when, sum as spark_sum, count, mean
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer
import numpy as np

# Initialiser Spark
spark = SparkSession.builder \
    .appName("FeatureImportanceWithoutVectorAssembler") \
    .getOrCreate()

# Créer un dataset synthétique
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
        (8.0, 9.0, 10.0, 12.0, 0),
        (9.0, 10.0, 11.0, 13.0, 1),
        (10.0, 11.0, 12.0, 14.0, 0)]

df = spark.createDataFrame(data, schema)

print("Dataset:")
df.show()

# Obtenir les noms des features
feature_columns = [col_name for col_name in df.columns if col_name != "label"]
print(f"Features: {feature_columns}")

# Méthode 1: Calcul de corrélation avec le label
print("\n=== MÉTHODE 1: CORRÉLATION AVEC LE LABEL ===")

correlation_results = []
for feature in feature_columns:
    # Calculer la corrélation de Pearson entre chaque feature et le label
    corr_df = df.select(feature, "label")
    corr_value = corr_df.stat.corr(feature, "label")
    correlation_results.append((feature, abs(corr_value)))

# Créer un DataFrame avec les corrélations
corr_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("correlation_abs", DoubleType(), True)
])

correlation_df = spark.createDataFrame(correlation_results, corr_schema)
correlation_sorted = correlation_df.orderBy(desc("correlation_abs"))

print("Importance basée sur la corrélation absolue:")
correlation_sorted.show()

# Méthode 2: Information Gain (approximation avec entropie)
print("\n=== MÉTHODE 2: INFORMATION GAIN (APPROXIMATION) ===")

# Calculer l'entropie du label
total_count = df.count()
label_counts = df.groupBy("label").count()

# Calculer l'entropie totale
entropy_components = label_counts.withColumn(
    "prob", col("count") / total_count
).withColumn(
    "entropy_part", -col("prob") * (col("prob").alias("log_prob"))
)

# Pour approximer le log, on utilise une transformation
# Note: PySpark n'a pas de fonction log directe dans SQL, donc on utilise une approximation
def calculate_entropy_gain(df, feature_col, label_col):
    """Calcule un proxy pour l'information gain"""
    
    # Discrétiser la feature continue en quartiles
    quantiles = df.approxQuantile(feature_col, [0.25, 0.5, 0.75], 0.01)
    
    # Créer des buckets
    df_buckets = df.withColumn(
        f"{feature_col}_bucket",
        when(col(feature_col) <= quantiles[0], 0)
        .when(col(feature_col) <= quantiles[1], 1)
        .when(col(feature_col) <= quantiles[2], 2)
        .otherwise(3)
    )
    
    # Calculer la pureté de chaque bucket
    bucket_purity = df_buckets.groupBy(f"{feature_col}_bucket").agg(
        count("*").alias("total_count"),
        spark_sum(when(col(label_col) == 1, 1).otherwise(0)).alias("positive_count")
    ).withColumn(
        "purity", 
        when(col("total_count") == 0, 0.0)
        .otherwise(abs(col("positive_count") / col("total_count") - 0.5) * 2)
    )
    
    # Calculer la pureté pondérée moyenne
    total_samples = df_buckets.count()
    weighted_purity = bucket_purity.withColumn(
        "weight", col("total_count") / total_samples
    ).withColumn(
        "weighted_purity", col("weight") * col("purity")
    ).agg(spark_sum("weighted_purity")).collect()[0][0]
    
    return weighted_purity if weighted_purity else 0.0

information_gain_results = []
for feature in feature_columns:
    gain = calculate_entropy_gain(df, feature, "label")
    information_gain_results.append((feature, gain))

# Créer DataFrame pour information gain
ig_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("information_gain", DoubleType(), True)
])

ig_df = spark.createDataFrame(information_gain_results, ig_schema)
ig_sorted = ig_df.orderBy(desc("information_gain"))

print("Importance basée sur l'Information Gain:")
ig_sorted.show()

# Méthode 3: Variance et séparation des classes
print("\n=== MÉTHODE 3: SÉPARATION DES CLASSES ===")

def calculate_class_separation(df, feature_col, label_col):
    """Calcule la différence entre les moyennes des deux classes"""
    
    stats_by_class = df.groupBy(label_col).agg(
        mean(feature_col).alias("mean_feature")
    )
    
    means = stats_by_class.select("mean_feature").collect()
    if len(means) >= 2:
        return abs(means[0][0] - means[1][0])
    return 0.0

separation_results = []
for feature in feature_columns:
    separation = calculate_class_separation(df, feature, "label")
    separation_results.append((feature, separation))

# Créer DataFrame pour séparation des classes
sep_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("class_separation", DoubleType(), True)
])

separation_df = spark.createDataFrame(separation_results, sep_schema)
separation_sorted = separation_df.orderBy(desc("class_separation"))

print("Importance basée sur la séparation des classes:")
separation_sorted.show()

# Méthode 4: Analyse univariée simple
print("\n=== MÉTHODE 4: ANALYSE UNIVARIÉE ===")

univariate_results = []
for feature in feature_columns:
    # Calculer statistiques par classe
    stats = df.groupBy("label").agg(
        mean(feature).alias("mean"),
        count("*").alias("count")
    ).collect()
    
    if len(stats) >= 2:
        # Calcul d'un score basé sur la différence des moyennes et taille des échantillons
        mean_diff = abs(stats[0]["mean"] - stats[1]["mean"])
        total_count = stats[0]["count"] + stats[1]["count"]
        score = mean_diff * np.sqrt(total_count)  # Pondérer par la taille
        univariate_results.append((feature, score))

# Créer DataFrame pour analyse univariée
univ_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("univariate_score", DoubleType(), True)
])

univariate_df = spark.createDataFrame(univariate_results, univ_schema)
univariate_sorted = univariate_df.orderBy(desc("univariate_score"))

print("Importance basée sur l'analyse univariée:")
univariate_sorted.show()

# Combiner toutes les méthodes
print("\n=== COMPARAISON DE TOUTES LES MÉTHODES ===")

# Joindre tous les résultats
final_comparison = correlation_sorted.alias("corr") \
    .join(ig_sorted.alias("ig"), col("corr.feature") == col("ig.feature")) \
    .join(separation_sorted.alias("sep"), col("corr.feature") == col("sep.feature")) \
    .join(univariate_sorted.alias("univ"), col("corr.feature") == col("univ.feature")) \
    .select(
        col("corr.feature").alias("feature"),
        col("corr.correlation_abs").alias("correlation"),
        col("ig.information_gain").alias("info_gain"),
        col("sep.class_separation").alias("separation"),
        col("univ.univariate_score").alias("univariate")
    )

print("Comparaison de toutes les méthodes:")
final_comparison.show()

# Calculer un score combiné (moyenne des rangs)
def add_rank_column(df, col_name, rank_col_name):
    window_spec = Window.orderBy(desc(col_name))
    return df.withColumn(rank_col_name, row_number().over(window_spec))

# Ajouter des rangs pour chaque méthode
ranked_corr = add_rank_column(correlation_sorted, "correlation_abs", "rank_corr")
ranked_ig = add_rank_column(ig_sorted, "information_gain", "rank_ig")
ranked_sep = add_rank_column(separation_sorted, "class_separation", "rank_sep")
ranked_univ = add_rank_column(univariate_sorted, "univariate_score", "rank_univ")

# Combiner les rangs
combined_ranks = ranked_corr.select("feature", "rank_corr") \
    .join(ranked_ig.select("feature", "rank_ig"), "feature") \
    .join(ranked_sep.select("feature", "rank_sep"), "feature") \
    .join(ranked_univ.select("feature", "rank_univ"), "feature") \
    .withColumn(
        "average_rank", 
        (col("rank_corr") + col("rank_ig") + col("rank_sep") + col("rank_univ")) / 4
    ).orderBy("average_rank")

print("Classement final basé sur la moyenne des rangs:")
combined_ranks.show()

# Afficher les statistiques descriptives des features
print("\n=== STATISTIQUES DESCRIPTIVES ===")
for feature in feature_columns:
    print(f"\nStatistiques pour {feature}:")
    df.select(feature).describe().show()

spark.stop()
```

Ce script modifié évite complètement l'utilisation de VectorAssembler et propose plusieurs méthodes alternatives pour évaluer l'importance des features :

1. **Corrélation avec le label** : Mesure la corrélation linéaire entre chaque feature et la variable cible

2. **Information Gain approximé** : Discrétise les features continues et calcule une approximation du gain d'information

3. **Séparation des classes** : Mesure la différence entre les moyennes des deux classes pour chaque feature

4. **Analyse univariée** : Combine la différence des moyennes avec la taille de l'échantillon

5. **Score combiné** : Calcule la moyenne des rangs de toutes les méthodes pour obtenir un classement final

Ces méthodes utilisent uniquement des opérations natives de PySpark DataFrame et fournissent une évaluation de l'importance des variables sans avoir recours aux algorithmes ML traditionnels qui nécessitent VectorAssembler.