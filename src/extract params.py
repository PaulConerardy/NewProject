Parfait, je vois les variations dans vos exemples. Voici une solution PySpark robuste qui gère ces différents formats :

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType
import re

def extract_info_robust(text):
    """
    Fonction robuste pour extraire les informations des différents formats de texte
    """
    
    # 1. Extraction du montant de l'opération
    # Gère "créditrice de X €" et "débitrice de X €"
    montant_patterns = [
        r"opération\s+(?:créditrice|débitrice)\s+de\s+([\d\s]+\.?\d*,?\d*)\s*€",
        r"montant\s+de\s*:\s*([\d\s]+\.?\d*,?\d*)\s*€"
    ]
    
    montant_op = None
    for pattern in montant_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            montant_op = match.group(1).replace(" ", "")
            break
    
    # 2. Extraction du pays
    # Gère "avec : Pays" et "avec le pays suivant : Pays"
    pays_patterns = [
        r"avec\s+le\s+pays\s+suivant\s*:\s*([^,]+)",
        r"avec\s*:\s*([^,\s]+(?:\s+[^,]+)*?)(?:\s+identifié|\s*,)"
    ]
    
    pays = None
    for pattern in pays_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pays = match.group(1).strip()
            break
    
    # 3. Extraction de la liste (rouge, orange, etc.)
    liste_match = re.search(r"liste\s+(\w+)", text, re.IGNORECASE)
    liste = liste_match.group(1) if liste_match else None
    
    # 4. Extraction du seuil
    # Gère différents formats de nombres
    seuil_patterns = [
        r"seuil\s+de\s+([\d\s]+\.?\d*,?\d*)\s*€",
        r"supérieur\s+au\s+seuil\s+de\s+([\d\s]+\.?\d*,?\d*)\s*€"
    ]
    
    seuil = None
    for pattern in seuil_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            seuil = match.group(1).replace(" ", "")
            break
    
    return (montant_op, pays, liste, seuil)

# Configuration Spark
spark = SparkSession.builder.appName("TextExtractionRobust").getOrCreate()

# Exemples de données avec les différents formats
data = [
    ("Le client a effectué une opération créditrice de 23 180,00 € avec le pays suivant : Lituanie , identifié sur la liste orange , dépassant le seuil de 20 000,00 € lors de la semaine.",),
    ("Le client a effectué une ou plusieurs opération(s) créditrice(s) avec : Philippines identifié sur la liste rouge d'un montant de : 32 000.00 €, supérieur au seuil de 29 000 €, sur les 5 dernières semaines.",),
    ("Le client a effectué une opération débitrice de 80 602.50 € avec le pays suivant : Afrique du Sud , identifié sur la liste rouge , dépassant le seuil de 40 000.00 € lors de la semaine.",)
]

df = spark.createDataFrame(data, ["texte"])

# Définir le schéma de retour
schema = StructType([
    StructField("montant_operation", StringType(), True),
    StructField("pays", StringType(), True),
    StructField("liste", StringType(), True),
    StructField("montant_seuil", StringType(), True)
])

# Créer l'UDF
extract_udf = udf(extract_info_robust, schema)

# Appliquer l'extraction
df_result = df.select(
    col("texte"),
    extract_udf(col("texte")).alias("extracted")
).select(
    col("texte"),
    col("extracted.montant_operation"),
    col("extracted.pays"),
    col("extracted.liste"),
    col("extracted.montant_seuil")
)

# Afficher les résultats
df_result.show(truncate=False)
```

**Version alternative avec regexp_extract pour plus de performance :**

```python
from pyspark.sql.functions import regexp_extract, coalesce, when, col

df_extracted = df.select(
    col("texte"),
    
    # Montant opération - plusieurs patterns possibles
    coalesce(
        regexp_extract(col("texte"), r"opération\s+(?:créditrice|débitrice)\s+de\s+([\d\s\.]+,?\d*)\s*€", 1),
        regexp_extract(col("texte"), r"montant\s+de\s*:\s*([\d\s\.]+,?\d*)\s*€", 1)
    ).alias("montant_operation_raw"),
    
    # Pays - plusieurs patterns
    coalesce(
        regexp_extract(col("texte"), r"avec\s+le\s+pays\s+suivant\s*:\s*([^,]+)", 1),
        regexp_extract(col("texte"), r"avec\s*:\s*([^\s]+(?:\s+[^,]+)*?)(?=\s+identifié)", 1)
    ).alias("pays"),
    
    # Liste
    regexp_extract(col("texte"), r"liste\s+(\w+)", 1).alias("liste"),
    
    # Seuil - plusieurs patterns
    coalesce(
        regexp_extract(col("texte"), r"seuil\s+de\s+([\d\s\.]+,?\d*)\s*€", 1),
        regexp_extract(col("texte"), r"supérieur\s+au\s+seuil\s+de\s+([\d\s\.]+,?\d*)\s*€", 1)
    ).alias("montant_seuil_raw")
).select(
    col("texte"),
    regexp_replace(col("montant_operation_raw"), r"\s+", "").alias("montant_operation"),
    trim(col("pays")).alias("pays"),
    col("liste"),
    regexp_replace(col("montant_seuil_raw"), r"\s+", "").alias("montant_seuil")
)
```

Cette solution gère :
- Les opérations créditrices ET débitrices
- Les différents formats de montants (avec points, virgules, espaces)
- Les variations dans la mention du pays ("avec :" vs "avec le pays suivant :")
- Les pays composés comme "Afrique du Sud"
- Les différentes formulations pour le seuil

**Résultats attendus :**
1. Exemple 1: montant="23180,00", pays="Lituanie", liste="orange", seuil="20000,00"
2. Exemple 2: montant="32000.00", pays="Philippines", liste="rouge", seuil="29000"
3. Exemple 3: montant="80602.50", pays="Afrique du Sud", liste="rouge", seuil="40000.00"