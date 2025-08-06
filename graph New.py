Voici comment adapter le code pour utiliser une variable catégorielle au lieu des bins de montants :

**Version de base avec variable catégorielle :**
```python
def plot_by_categorical_variable(df, rule_id, categorical_col, percentage_statuses=['Concluante', 'A revoir']):
    """
    Créer un histogramme empilé avec ligne de pourcentage selon une variable catégorielle
    
    Args:
        df: DataFrame PySpark
        rule_id: ID de la règle à analyser
        categorical_col: nom de la colonne catégorielle (ex: 'REGION', 'TYPE_CLIENT', etc.)
        percentage_statuses: liste des statuts considérés comme "positifs" pour le calcul du pourcentage
    """
    
    print(f"Traitement de la règle {rule_id} selon la variable '{categorical_col}'...")
    
    # Filtrer pour la règle spécifique
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    # Vérifier qu'on a des données
    count_rule = df_rule.count()
    if count_rule == 0:
        print(f"Aucune donnée trouvée pour RULE_ID = {rule_id}")
        return
    
    print(f"Nombre de lignes pour cette règle: {count_rule}")
    
    # Vérifier les valeurs de la variable catégorielle
    print(f"\n=== Valeurs uniques de {categorical_col} ===")
    df_rule.select(categorical_col).distinct().orderBy(categorical_col).show()
    
    # Nettoyer les valeurs nulles de la variable catégorielle
    df_rule = df_rule.withColumn(
        f"{categorical_col}_clean",
        when(col(categorical_col).isNull(), "Non renseigné").otherwise(col(categorical_col))
    )
    
    # Grouper par variable catégorielle et statut
    counts = df_rule.groupBy(f"{categorical_col}_clean", "STATUS_NAME").agg(
        count("*").alias("count")
    ).orderBy(f"{categorical_col}_clean")
    
    print(f"\n=== Distribution par {categorical_col} et statut ===")
    counts.show()
    
    # Convertir en pandas
    counts_pd = counts.toPandas()
    
    if counts_pd.empty:
        print("DataFrame pandas vide")
        return
    
    # Créer le pivot
    pivot_data = counts_pd.pivot(index=f"{categorical_col}_clean", columns='STATUS_NAME', values='count').fillna(0)
    
    print(f"\n=== Données pivotées ===")
    print(pivot_data)
    
    # Calculer les pourcentages
    percentage_cols = [col for col in pivot_data.columns if col in percentage_statuses]
    
    if percentage_cols:
        percentage_series = pivot_data[percentage_cols].sum(axis=1)
    else:
        percentage_series = pd.Series(0, index=pivot_data.index)
    
    total_series = pivot_data.sum(axis=1)
    percentage_rate = (percentage_series / total_series * 100).fillna(0)
    
    # Créer le graphique avec deux axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Couleurs pour les statuts
    available_colors = {'Non concluante': 'grey', 'Concluante': 'green', 'A revoir': 'darkblue'}
    colors = [available_colors.get(col, 'blue') for col in pivot_data.columns]
    
    # Histogramme empilé
    pivot_data.plot(kind='bar', stacked=True, 
                    color=colors[:len(pivot_data.columns)],
                    ax=ax1, width=0.7)
    
    ax1.set_xlabel(f'{categorical_col}', fontsize=12)
    ax1.set_ylabel('Nombre d\'alertes', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Second axe pour les pourcentages
    ax2 = ax1.twinx()
    
    # Ligne des pourcentages
    x_positions = range(len(percentage_rate))
    ax2.plot(x_positions, percentage_rate.values, 
             color='red', marker='o', linewidth=3, markersize=8,
             label=f'% {"/".join(percentage_statuses)}')
    
    ax2.set_ylabel('Pourcentage (%)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Annotations sur la ligne
    for i, (x, y) in enumerate(zip(x_positions, percentage_rate.values)):
        if total_series.iloc[i] > 0:
            ax2.annotate(f'{y:.1f}%', 
                        xy=(x, y), 
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        color='red', fontweight='bold', fontsize=9)
    
    # Titre et légendes
    plt.title(f'Distribution des statuts par {categorical_col} - Règle {rule_id}', 
              fontsize=14, fontweight='bold')
    
    ax1.legend(loc='upper left', title='Statut')
    ax2.legend(loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Afficher le résumé
    print(f"\n=== Résumé par {categorical_col} ===")
    summary_df = pd.DataFrame({
        'Total': total_series,
        f'{"/".join(percentage_statuses)}': percentage_series,
        'Pourcentage': percentage_rate.round(1)
    })
    print(summary_df)

# Utilisation
plot_by_categorical_variable(your_df, your_rule_id, 'REGION')
plot_by_categorical_variable(your_df, your_rule_id, 'TYPE_CLIENT')
plot_by_categorical_variable(your_df, your_rule_id, 'CHANNEL')
```

**Version pour analyser plusieurs variables catégorielles d'un coup :**
```python
def plot_multiple_categorical_analysis(df, rule_id, categorical_cols, percentage_statuses=['Concluante', 'A revoir']):
    """
    Analyser plusieurs variables catégorielles et créer un subplot pour chacune
    """
    
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 1) // 2  # Arrondir vers le haut pour 2 colonnes
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, categorical_col in enumerate(categorical_cols):
        
        # Filtrer et préparer les données
        df_rule = df.filter(col("RULE_ID") == rule_id)
        df_rule = df_rule.withColumn(
            f"{categorical_col}_clean",
            when(col(categorical_col).isNull(), "Non renseigné").otherwise(col(categorical_col))
        )
        
        # Grouper et pivoter
        counts = df_rule.groupBy(f"{categorical_col}_clean", "STATUS_NAME").agg(
            count("*").alias("count")
        ).orderBy(f"{categorical_col}_clean")
        
        counts_pd = counts.toPandas()
        
        if counts_pd.empty:
            continue
            
        pivot_data = counts_pd.pivot(index=f"{categorical_col}_clean", columns='STATUS_NAME', values='count').fillna(0)
        
        # Calculer pourcentages
        percentage_cols = [col for col in pivot_data.columns if col in percentage_statuses]
        percentage_series = pivot_data[percentage_cols].sum(axis=1) if percentage_cols else pd.Series(0, index=pivot_data.index)
        total_series = pivot_data.sum(axis=1)
        percentage_rate = (percentage_series / total_series * 100).fillna(0)
        
        # Graphique
        ax1 = axes[i]
        
        colors = ['grey', 'green', 'darkblue'][:len(pivot_data.columns)]
        pivot_data.plot(kind='bar', stacked=True, color=colors, ax=ax1, width=0.7)
        
        ax2 = ax1.twinx()
        x_positions = range(len(percentage_rate))
        ax2.plot(x_positions, percentage_rate.values, 
                color='red', marker='o', linewidth=2, markersize=6)
        
        # Styling
        ax1.set_title(f'{categorical_col} - Règle {rule_id}', fontsize=12, fontweight='bold')
        ax1.set_xlabel(categorical_col, fontsize=10)
        ax1.set_ylabel('Nombre', fontsize=10)
        ax2.set_ylabel('% Concluance', color='red', fontsize=10)
        ax2.set_ylim(0, 105)
        
        # Annotations
        for x, y in zip(x_positions, percentage_rate.values):
            if total_series.iloc[x] > 0:
                ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                           textcoords='offset points', ha='center', va='bottom',
                           color='red', fontsize=8)
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(fontsize=8)
    
    # Cacher les axes vides s'il y en a
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Utilisation
categorical_variables = ['REGION', 'TYPE_CLIENT', 'CHANNEL', 'PRODUCT_TYPE']
plot_multiple_categorical_analysis(your_df, your_rule_id, categorical_variables)
```

**Version simplifiée pour exploration rapide :**
```python
def quick_categorical_analysis(df, rule_id, categorical_col):
    """Version rapide pour exploration"""
    
    df_rule = df.filter(col("RULE_ID") == rule_id)
    
    # Simple groupBy
    result = df_rule.groupBy(categorical_col, "STATUS_NAME").agg(
        count("*").alias("count")
    ).orderBy(categorical_col)
    
    # Affichage PySpark direct
    print(f"=== Distribution pour {categorical_col} - Règle {rule_id} ===")
    result.show()
    
    # Version graphique simple
    result_pd = result.toPandas()
    if not result_pd.empty:
        pivot = result_pd.pivot(index=categorical_col, columns='STATUS_NAME', values='count').fillna(0)
        
        pivot.plot(kind='bar', stacked=True, figsize=(10, 6),
                  color=['grey', 'green', 'darkblue'][:len(pivot.columns)])
        
        plt.title(f'Distribution par {categorical_col} - Règle {rule_id}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Utilisation rapide
quick_categorical_analysis(your_df, your_rule_id, 'REGION')
```

Ces versions vous permettent d'analyser n'importe quelle variable catégorielle au lieu des bins de montants, en conservant la même logique de visualisation avec histogramme empilé et ligne de pourcentage.