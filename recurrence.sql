from pyspark.sql.functions import col, count, collect_set, array_sort, array_join, sum

# Étape 1 : Agréger les règles uniques et compter les alertes par client
client_summary_df = df_alertes.groupBy("id_client").agg(
    # collect_set récupère les valeurs uniques dans un tableau
    collect_set("règle").alias("set_de_regles"),
    count("id_alerte").alias("total_alertes_client")
)

# Étape 2 : Créer la chaîne de combinaison canonique (triée)
client_combinations_df = client_summary_df.withColumn(
    "combinaison_regles",
    # On trie le tableau de règles, puis on le transforme en chaîne
    array_join(array_sort(col("set_de_regles")), "/")
)

# Étape 3 : Regrouper par la chaîne de combinaison et calculer les métriques finales
result_df = client_combinations_df.groupBy("combinaison_regles").agg(
    count("id_client").alias("nombre_clients_uniques"),
    sum("total_alertes_client").alias("nombre_total_alertes")
).orderBy(col("nombre_clients_uniques").desc())

result_df.show(truncate=False)
