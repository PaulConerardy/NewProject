Le problème vient probablement de la logique des bins ou du filtrage des données. Voici comment diagnostiquer et corriger :

**1. D'abord, diagnostiquons le problème :**
```python
def debug_recurrence_data(df, rule_id):
    """Fonction pour diagnostiquer les données"""
    
    # Vérifier les données de base
    print(f"=== DEBUG pour la règle {rule_id} ===")
    
    # Nombre total de lignes
    total_rows = df.count()
    print(f"Nombre total de lignes dans le df: {total_rows}")
    
    # Vérifier si la règle existe
    rule_exists = df.filter(col("RULE_ID") == rule_id).count()
    print(f"Nombre de lignes pour RULE_ID {rule_id}: {rule_exists}")
    
    # Vérifier les valeurs de recurrence_retro
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    print("\n=== Statistiques recurrence_retro ===")
    df_rule.select("recurrence_retro").describe().show()
    
    print("\n=== Valeurs uniques de recurrence_retro ===")
    df_rule.select("recurrence_retro").distinct().orderBy("recurrence_retro").show()
    
    print("\n=== Valeurs uniques de STATUS_NAME ===")
    df_rule.select("STATUS_NAME").distinct().show()
    
    print("\n=== Distribution STATUS_NAME ===")
    df_rule.groupBy("STATUS_NAME").count().show()
    
    return df_rule

# Utilisez cette fonction d'abord
df_debug = debug_recurrence_data(your_df, your_rule_id)
```

**2. Version corrigée de la fonction avec gestion des cas vides :**
```python
def plot_recurrence_histogram_fixed(df, rule_id):
    """Version corrigée avec vérifications"""
    
    print(f"Traitement de la règle {rule_id}...")
    
    # Filtrer pour la règle spécifique
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    # Vérifier qu'on a des données
    count_rule = df_rule.count()
    if count_rule == 0:
        print(f"Aucune donnée trouvée pour RULE_ID = {rule_id}")
        return
    
    print(f"Nombre de lignes pour cette règle: {count_rule}")
    
    # Nettoyer les valeurs nulles
    df_rule = df_rule.withColumn(
        "recurrence_clean",
        when(col("recurrence_retro").isNull(), 0).otherwise(col("recurrence_retro"))
    )
    
    # Créer des bins plus simples d'abord
    df_rule = df_rule.withColumn(
        "recurrence_bin",
        when(col("recurrence_clean") == 0, "0")
        .when(col("recurrence_clean") == 1, "1")
        .when(col("recurrence_clean").between(2, 5), "2-5") 
        .when(col("recurrence_clean").between(6, 10), "6-10")
        .otherwise("10+")
    )
    
    # Vérifier la distribution des bins
    print("\n=== Distribution des bins ===")
    df_rule.groupBy("recurrence_bin").count().orderBy("recurrence_bin").show()
    
    # Grouper par bin et statut
    counts = df_rule.groupBy("recurrence_bin", "STATUS_NAME").agg(
        count("*").alias("count")
    ).orderBy("recurrence_bin")
    
    # Vérifier qu'on a des résultats
    counts_collected = counts.collect()
    if len(counts_collected) == 0:
        print("Aucun résultat après groupBy")
        return
    
    print(f"\n=== Résultats groupBy ({len(counts_collected)} lignes) ===")
    counts.show()
    
    # Convertir en pandas
    counts_pd = counts.toPandas()
    
    if counts_pd.empty:
        print("DataFrame pandas vide")
        return
    
    print(f"\n=== DataFrame pandas ===")
    print(counts_pd)
    
    # Créer le pivot
    pivot_data = counts_pd.pivot(index='recurrence_bin', columns='STATUS_NAME', values='count').fillna(0)
    
    print(f"\n=== Données pivotées ===")
    print(pivot_data)
    
    # Vérifier qu'on a des données non nulles
    if pivot_data.sum().sum() == 0:
        print("Toutes les valeurs sont à 0")
        return
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Utiliser les couleurs disponibles selon les colonnes présentes
    available_colors = {'Non concluante': 'grey', 'Concluante': 'green', 'A revoir': 'darkblue'}
    colors = [available_colors.get(col, 'blue') for col in pivot_data.columns]
    
    pivot_data.plot(kind='bar', stacked=True, 
                    color=colors[:len(pivot_data.columns)],
                    ax=ax)
    
    plt.title(f'Distribution des statuts par récurrence - Règle {rule_id}')
    plt.xlabel('Bins de récurrence')
    plt.ylabel('Nombre d\'alertes')
    plt.legend(title='Statut')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Utilisation
plot_recurrence_histogram_fixed(your_df, your_rule_id)
```

**3. Version encore plus simple pour tester :**
```python
def test_simple_plot(df, rule_id):
    """Test très simple"""
    
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    # Just count by STATUS_NAME, ignore bins for now
    simple_counts = df_rule.groupBy("STATUS_NAME").count().toPandas()
    
    if not simple_counts.empty:
        simple_counts.set_index('STATUS_NAME')['count'].plot(kind='bar')
        plt.title(f'Simple count by status - Rule {rule_id}')
        plt.show()
    else:
        print("No data found")

# Test d'abord avec cette version
test_simple_plot(your_df, your_rule_id)
```

Commencez par la fonction `debug_recurrence_data` pour voir ce qui se passe avec vos données, puis utilisez la version corrigée.