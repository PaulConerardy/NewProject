Je vois que vous avez déjà un code en Python/pandas pour créer des histogrammes de récurrence par statut. Voici comment adapter ce code à PySpark :

```python
from pyspark.sql.functions import col, when, max as spark_max, count, desc
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
import pandas as pd

def plot_recurrence_line_pyspark(dataframe, rules, population, 
                                bins=None, bin_labels=None, 
                                bar_width=0.3, **kwargs):
    """
    Histogrammes de la récurrence par nombre d'alertes - Version PySpark
    """
    
    # Copie pour ne pas altérer la table en dehors de cette fonction
    df_test = dataframe
    
    # Restriction à la population choisie et aux colonnes souhaitées
    if population:
        df_test = df_test.filter(col("population_short").isin(population))
    
    df_test = df_test.select("ALERT_ID", "RULE_ID", "STATUS_NAME", "recurrence_retro", "E_STATE")
    
    # Supprimer les lignes identiques (équivalent de drop_duplicates)
    df_test = df_test.distinct()
    
    # Remplacer les valeurs nulles par 0 dans recurrence_retro
    df_test = df_test.withColumn(
        "recurrence_retro", 
        when(col("recurrence_retro").isNull(), 0).otherwise(col("recurrence_retro"))
    )
    
    # Pour chaque règle, produire plusieurs graphiques selon la finalité
    for rule in rules:
        df_test_rule = df_test.filter(col("RULE_ID") == rule)
        
        # Regrouper la récurrence par intervalle (bin) de valeurs
        max_recurrence = df_test_rule.agg(spark_max("recurrence_retro")).collect()[0][0]
        
        if bins is None:
            bins = list(range(1, min(31, max_recurrence + 2)))
        
        if bin_labels is None:
            bin_labels = [f"{i}-{i+1}" if i < 10 else "10-14" if i == 10 else "15+" 
                         for i in range(1, len(bins))]
        
        # Fonction pour assigner les bins (équivalent de pd.cut)
        def assign_bin(recurrence):
            if recurrence is None:
                return "0"
            for i, bin_val in enumerate(bins[1:]):
                if recurrence <= bin_val:
                    return bin_labels[i] if i < len(bin_labels) else bin_labels[-1]
            return bin_labels[-1]
        
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType
        assign_bin_udf = udf(assign_bin, StringType())
        
        df_test_rule = df_test_rule.withColumn("bins", assign_bin_udf(col("recurrence_retro")))
        
        # Fréquence par intervalle pour chaque type d'alerte
        # Non concluantes
        non_concl = df_test_rule.filter(col("STATUS_NAME") == "Non concluante")
        frequence_non_concluantes = non_concl.groupBy("bins").agg(count("*").alias("count")).orderBy("bins")
        
        # Concluantes  
        concl = df_test_rule.filter(col("STATUS_NAME") == "Concluante")
        frequence_concluantes = concl.groupBy("bins").agg(count("*").alias("count")).orderBy("bins")
        
        # À revoir
        a_revoir = df_test_rule.filter(col("STATUS_NAME") == "A revoir")
        frequence_a_revoir = a_revoir.groupBy("bins").agg(count("*").alias("count")).orderBy("bins")
        
        # Convertir en dictionnaires pour la visualisation
        def spark_to_dict(spark_df, bin_labels):
            pandas_df = spark_df.toPandas()
            result_dict = {label: 0 for label in bin_labels}
            for row in pandas_df.itertuples():
                if row.bins in result_dict:
                    result_dict[row.bins] = row.count
            return result_dict
        
        bins_dict = {label: i for i, label in enumerate(bin_labels)}
        
        # Convertir les résultats
        freq_non_concl_dict = spark_to_dict(frequence_non_concluantes, bin_labels)
        freq_concl_dict = spark_to_dict(frequence_concluantes, bin_labels) 
        freq_revoir_dict = spark_to_dict(frequence_a_revoir, bin_labels)
        
        # Créer les séries pandas pour la visualisation
        frequence_non_concluantes_series = pd.Series(freq_non_concl_dict)
        frequence_concluantes_series = pd.Series(freq_concl_dict)
        frequence_a_revoir_series = pd.Series(freq_revoir_dict)
        
        # Création des graphiques
        fig = plt.figure(**kwargs)
        
        # Premier graphique : Fréquence de la récurrence parmi les alertes non-concluantes
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel("Nombre d'alertes se retrouvant dans chaque intervalle")
        
        frequence_non_concluantes_series.plot(kind="bar", color="grey", ax=ax1, 
                                            rot=0, width=bar_width,
                                            label="Non concluantes")
        
        frequence_concluantes_series.plot(kind="bar", color="green", ax=ax1, bottom=frequence_non_concluantes_series,
                                        rot=0, width=bar_width, 
                                        label="Concluantes")
        
        frequence_a_revoir_series.plot(kind="bar", color="darkblue", ax=ax1, 
                                     bottom=frequence_non_concluantes_series + frequence_concluantes_series,
                                     rot=0, width=bar_width,
                                     label="Ouvertes (à revoir)")
        
        # Ajouter les valeurs de concluance sur les barres
        concluance_series = frequence_concluantes_series + frequence_a_revoir_series
        total_series = frequence_non_concluantes_series + frequence_concluantes_series + frequence_a_revoir_series
        
        for i, (concl_val, total_val) in enumerate(zip(concluance_series, total_series)):
            if total_val > 0:
                percentage = (concl_val / total_val) * 100
                ax1.text(i, total_val + 0.5, f"{percentage:.1f}%", 
                        ha='center', va='bottom', fontsize=8)
        
        ax1.legend()
        ax1.set_title(f"Règle {rule} - Distribution par statut et récurrence")
        
        plt.tight_layout()
        plt.show()

# Utilisation
# plot_recurrence_line_pyspark(your_spark_df, rules=[123, 456], population=["pop1", "pop2"])
```

**Version simplifiée si vous voulez juste l'histogramme empilé :**

```python
def simple_recurrence_histogram_pyspark(df, rule_id):
    """Version simplifiée pour un histogramme empilé"""
    
    # Filtrer pour la règle spécifique
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    # Créer les bins de récurrence
    df_rule = df_rule.withColumn(
        "recurrence_bin",
        when(col("recurrence_retro") <= 1, "1")
        .when(col("recurrence_retro") <= 5, "2-5") 
        .when(col("recurrence_retro") <= 10, "6-10")
        .otherwise("10+")
    )
    
    # Grouper par bin et statut
    counts = df_rule.groupBy("recurrence_bin", "STATUS_NAME").agg(
        count("*").alias("count")
    ).orderBy("recurrence_bin")
    
    # Convertir en pandas et créer le graphique
    counts_pd = counts.toPandas()
    pivot_data = counts_pd.pivot(index='recurrence_bin', columns='STATUS_NAME', values='count').fillna(0)
    
    # Graphique empilé
    ax = pivot_data.plot(kind='bar', stacked=True, 
                        color=['grey', 'green', 'darkblue'],
                        figsize=(10, 6))
    
    plt.title(f'Distribution des statuts par récurrence - Règle {rule_id}')
    plt.xlabel('Nombre de récurrences')
    plt.ylabel('Nombre d\'alertes')
    plt.legend(title='Statut')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Utilisation
# simple_recurrence_histogram_pyspark(your_spark_df, 123)
```

Cette adaptation conserve la logique de votre code original tout en utilisant les fonctions PySpark pour le traitement des données.