Voici comment ajouter un second axe avec une ligne pour les pourcentages :

```python
def plot_recurrence_histogram_with_percentage_line(df, rule_id):
    """Version avec ligne de pourcentage sur second axe"""
    
    print(f"Traitement de la règle {rule_id}...")
    
    # Votre code existant jusqu'à la création du pivot_data
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    df_rule = df_rule.withColumn(
        "recurrence_clean",
        when(col("recurrence_retro").isNull(), 0).otherwise(col("recurrence_retro"))
    )
    
    df_rule = df_rule.withColumn(
        "recurrence_bin",
        when(col("recurrence_clean") == 0, "0")
        .when(col("recurrence_clean") == 1, "1")
        .when(col("recurrence_clean").between(2, 5), "2-5") 
        .when(col("recurrence_clean").between(6, 10), "6-10")
        .otherwise("10+")
    )
    
    counts = df_rule.groupBy("recurrence_bin", "STATUS_NAME").agg(
        count("*").alias("count")
    ).orderBy("recurrence_bin")
    
    counts_pd = counts.toPandas()
    pivot_data = counts_pd.pivot(index='recurrence_bin', columns='STATUS_NAME', values='count').fillna(0)
    
    # Calculer les pourcentages de concluance
    # Supposons que "Concluante" et "A revoir" sont considérées comme concluantes
    concluance_cols = [col for col in pivot_data.columns if col in ['Concluante', 'A revoir']]
    
    if concluance_cols:
        concluance_series = pivot_data[concluance_cols].sum(axis=1)
    else:
        # Si pas de colonnes de concluance, créer une série vide
        concluance_series = pd.Series(0, index=pivot_data.index)
    
    total_series = pivot_data.sum(axis=1)
    
    # Calculer les pourcentages (éviter division par zéro)
    percentage_series = (concluance_series / total_series * 100).fillna(0)
    
    # Créer le graphique avec deux axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Premier axe : histogramme empilé
    available_colors = {'Non concluante': 'grey', 'Concluante': 'green', 'A revoir': 'darkblue'}
    colors = [available_colors.get(col, 'blue') for col in pivot_data.columns]
    
    pivot_data.plot(kind='bar', stacked=True, 
                    color=colors[:len(pivot_data.columns)],
                    ax=ax1, width=0.6)
    
    ax1.set_xlabel('Bins de récurrence')
    ax1.set_ylabel('Nombre d\'alertes', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title(f'Distribution des statuts par récurrence - Règle {rule_id}')
    
    # Créer le second axe pour les pourcentages
    ax2 = ax1.twinx()
    
    # Tracer la ligne des pourcentages
    x_positions = range(len(percentage_series))
    ax2.plot(x_positions, percentage_series.values, 
             color='red', marker='o', linewidth=2, markersize=6,
             label='% Concluance')
    
    ax2.set_ylabel('Pourcentage de concluance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)  # Pourcentages de 0 à 100
    
    # Ajouter les valeurs sur la ligne
    for i, (x, y) in enumerate(zip(x_positions, percentage_series.values)):
        if total_series.iloc[i] > 0:  # Seulement si on a des données
            ax2.annotate(f'{y:.1f}%', 
                        xy=(x, y), 
                        xytext=(0, 10),  # 10 points vers le haut
                        textcoords='offset points',
                        ha='center', va='bottom',
                        color='red', fontweight='bold')
    
    # Légendes
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    # Ajuster les labels de l'axe x
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Optionnel : afficher les données numériques
    print("\n=== Résumé des données ===")
    summary_df = pd.DataFrame({
        'Total': total_series,
        'Concluantes': concluance_series,
        'Pourcentage': percentage_series
    })
    print(summary_df)

# Utilisation
plot_recurrence_histogram_with_percentage_line(your_df, your_rule_id)
```

**Version alternative si vous voulez plus de contrôle sur le calcul des pourcentages :**

```python
def plot_with_custom_percentage_calculation(df, rule_id, concluant_statuses=['Concluante', 'A revoir']):
    """Version avec calcul personnalisé des pourcentages"""
    
    # ... votre code existant jusqu'au pivot_data ...
    
    # Calculer les séries selon votre logique métier
    if any(status in pivot_data.columns for status in concluant_statuses):
        available_concluant_cols = [col for col in concluant_statuses if col in pivot_data.columns]
        concluance_series = pivot_data[available_concluant_cols].sum(axis=1)
    else:
        concluance_series = pd.Series(0, index=pivot_data.index)
    
    total_series = pivot_data.sum(axis=1)
    
    # Votre calcul spécifique (par exemple, différent de ce qui était dans le code original)
    percentage_series = ((concluance_series / total_series) * 100).round(1)
    
    # Créer le graphique
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Histogramme
    pivot_data.plot(kind='bar', stacked=True, ax=ax1, width=0.7,
                    color=['grey', 'green', 'darkblue'][:len(pivot_data.columns)])
    
    # Second axe pour la ligne
    ax2 = ax1.twinx()
    
    # Ligne des pourcentages avec style amélioré
    line = ax2.plot(range(len(percentage_series)), percentage_series.values,
                    color='red', marker='D', linewidth=3, markersize=8,
                    label='Taux de concluance (%)', linestyle='-', alpha=0.8)
    
    # Styling des axes
    ax1.set_ylabel('Nombre d\'alertes', fontsize=12, color='black')
    ax2.set_ylabel('Taux de concluance (%)', fontsize=12, color='red')
    ax1.set_xlabel('Bins de récurrence', fontsize=12)
    
    # Limites et grille
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Annotations sur la ligne
    for i, (x, y) in enumerate(zip(range(len(percentage_series)), percentage_series.values)):
        if total_series.iloc[i] > 0:
            ax2.annotate(f'{y:.1f}%', 
                        xy=(x, y), 
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        color='red', fontweight='bold', fontsize=10)
    
    # Titre et légendes
    plt.title(f'Distribution des statuts et taux de concluance - Règle {rule_id}', 
              fontsize=14, fontweight='bold')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Utilisation
plot_with_custom_percentage_calculation(your_df, your_rule_id, 
                                       concluant_statuses=['Concluante', 'A revoir'])
```

Cette solution vous donne :
1. Un histogramme empilé sur l'axe principal (gauche)
2. Une ligne avec les pourcentages sur l'axe secondaire (droite)
3. Les valeurs des pourcentages affichées au-dessus de chaque point
4. Des légendes séparées pour chaque élément
5. Une grille pour faciliter la lecture des pourcentages