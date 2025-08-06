from pyspark.sql.functions import col, count, collect_set, array_sort, array_join, sum, round, when, lit

# --- ÉTAPE 0 : Définir le statut que vous voulez analyser ---
# Remplacez cette valeur par le statut exact présent dans vos données
statut_cible = "Cloturé - Faux Positif"


# Étape 1 : Agréger par client en comptant le total ET les alertes avec le statut cible
client_summary_df = df_alertes.groupBy("id_client").agg(
    collect_set("règle").alias("set_de_regles"),
    count("id_alerte").alias("total_alertes_client"),
    
    # NOUVEAU : Comptage conditionnel pour le statut cible
    sum(when(col("statut") == statut_cible, 1).otherwise(0)).alias("alertes_cible_client")
)

# Étape 2 : Créer la chaîne de combinaison canonique (triée)
client_combinations_df = client_summary_df.withColumn(
    "combinaison_regles",
    array_join(array_sort(col("set_de_regles")), "/")
)

# Étape 3 : Regrouper par combinaison et sommer tous les compteurs
# On somme maintenant le total des alertes ET le total des alertes cibles
result_df = client_combinations_df.groupBy("combinaison_regles").agg(
    count("id_client").alias("nombre_clients_uniques"),
    sum("total_alertes_client").alias("nombre_total_alertes"),
    sum("alertes_cible_client").alias("nombre_total_alertes_cible") # NOUVEAU
)

# Étape 4 : Calculer les ratios finaux et les formater
final_result_df = result_df.withColumn(
    "alertes_moyennes_par_client",
    round(
        col("nombre_total_alertes") / col("nombre_clients_uniques"), 
        2
    )
).withColumn(
    # NOUVELLE COLONNE : Calcul du pourcentage du statut cible
    f"pourcentage_{statut_cible.lower().replace(' ', '_')}", # Crée un nom de colonne dynamique, ex: 'pourcentage_cloture_faux_positif'
    round(
        (col("nombre_total_alertes_cible") / col("nombre_total_alertes")) * 100, 
        2
    )
).orderBy(col("nombre_clients_uniques").desc())


# Afficher le résultat final
final_result_df.show(truncate=False)

