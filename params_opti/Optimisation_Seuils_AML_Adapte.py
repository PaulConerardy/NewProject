import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Définir le style des graphiques
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Charger les données générées
chemin_donnees = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df = pd.read_csv(chemin_donnees)

# Charger les seuils
chemin_excel = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
seuils = {}
for regle in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    seuils[regle] = pd.read_excel(chemin_excel, sheet_name=regle)

# Afficher les informations de base sur le jeu de données
print(f"Forme du jeu de données: {df.shape}")
print("\nÉchantillon de données:")
print(df.head())

# Créer un id_alerte pour regrouper par alerte
df['id_alerte'] = df['alert_date'] + '_' + df['account_number'].astype(str)

# Analyser la distribution des alertes
comptage_alertes = df.groupby('id_alerte')['is_issue'].max().reset_index()
total_alertes = len(comptage_alertes)
vrais_positifs = comptage_alertes['is_issue'].sum()
faux_positifs = total_alertes - vrais_positifs

print(f"\nTotal des alertes: {total_alertes}")
print(f"Vrais positifs: {vrais_positifs} ({vrais_positifs/total_alertes:.2%})")
print(f"Faux positifs: {faux_positifs} ({faux_positifs/total_alertes:.2%})")

# Analyser la distribution des paramètres
print("\nDistribution des paramètres:")
comptage_params = df['param'].value_counts()
print(comptage_params.head(10))

# Fonction pour évaluer la performance d'un seuil
def evaluer_seuil(id_regle, segment, parametre, valeur_seuil):
    """
    Évaluer la performance d'une valeur de seuil spécifique pour un paramètre
    """
    # Filtrer les données pour cette règle-segment-paramètre
    donnees_param = df[(df['rule_id'] == id_regle) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parametre)]
    
    if donnees_param.empty:
        return {
            'seuil': valeur_seuil,
            'vp': 0, 'fp': 0, 'vn': 0, 'fn': 0,
            'precision': 0, 'rappel': 0, 'f1': 0, 'specificite': 0
        }
    
    # Regrouper par id_alerte pour éviter le double comptage
    alertes = donnees_param.groupby('id_alerte')['is_issue'].max().reset_index()
    alertes['predit'] = 0
    
    # Appliquer le seuil
    for id_alerte in alertes['id_alerte']:
        donnees_alerte = donnees_param[donnees_param['id_alerte'] == id_alerte]
        if any(donnees_alerte['value'] >= valeur_seuil):
            alertes.loc[alertes['id_alerte'] == id_alerte, 'predit'] = 1
    
    # Calculer les métriques
    vp = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 1))
    fp = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 1))
    vn = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 0))
    fn = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 0))
    
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
    specificite = vn / (vn + fp) if (vn + fp) > 0 else 0
    
    return {
        'seuil': valeur_seuil,
        'vp': vp, 'fp': fp, 'vn': vn, 'fn': fn,
        'precision': precision, 'rappel': rappel, 'f1': f1, 'specificite': specificite
    }

# Fonction pour trouver le seuil optimal pour un paramètre
def trouver_seuil_optimal(id_regle, segment, parametre, metrique='f1'):
    """
    Trouver la valeur de seuil optimale pour un paramètre basé sur une métrique spécifiée
    """
    # Filtrer les données pour cette règle-segment-paramètre
    donnees_param = df[(df['rule_id'] == id_regle) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parametre)]
    
    if donnees_param.empty:
        print(f"Pas de données pour {id_regle}-{segment}-{parametre}")
        return None, None
    
    # Obtenir les valeurs uniques et les trier
    valeurs = sorted(donnees_param['value'].unique())
    
    # Si trop de valeurs, échantillonner un nombre raisonnable
    if len(valeurs) > 50:
        valeurs = np.percentile(donnees_param['value'], np.linspace(0, 100, 50))
    
    # S'assurer que toutes les valeurs sont des entiers
    valeurs = [int(round(v)) for v in valeurs]
    valeurs = sorted(list(set(valeurs)))  # Supprimer les doublons
    
    # Évaluer chaque seuil
    resultats = []
    for seuil in valeurs:
        metriques = evaluer_seuil(id_regle, segment, parametre, seuil)
        resultats.append(metriques)
    
    # Convertir en DataFrame
    df_resultats = pd.DataFrame(resultats)
    
    # Trouver le seuil optimal basé sur la métrique spécifiée
    if not df_resultats.empty:
        idx_optimal = df_resultats[metrique].idxmax()
        seuil_optimal = df_resultats.loc[idx_optimal, 'seuil']
        metriques_optimales = df_resultats.loc[idx_optimal]
        
        return seuil_optimal, metriques_optimales
    
    return None, None

# Fonction pour tracer les courbes de performance des seuils
def tracer_courbes_seuil(id_regle, segment, parametre):
    """
    Tracer les courbes de performance pour différentes valeurs de seuil
    """
    # Filtrer les données pour cette règle-segment-paramètre
    donnees_param = df[(df['rule_id'] == id_regle) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parametre)]
    
    if donnees_param.empty:
        print(f"Pas de données pour {id_regle}-{segment}-{parametre}")
        return
    
    # Obtenir les valeurs uniques et les trier
    valeurs = sorted(donnees_param['value'].unique())
    
    # Si trop de valeurs, échantillonner un nombre raisonnable
    if len(valeurs) > 50:
        valeurs = np.percentile(donnees_param['value'], np.linspace(0, 100, 50))
    
    # S'assurer que toutes les valeurs sont des entiers
    valeurs = [int(round(v)) for v in valeurs]
    valeurs = sorted(list(set(valeurs)))  # Supprimer les doublons
    
    # Évaluer chaque seuil
    resultats = []
    for seuil in valeurs:
        metriques = evaluer_seuil(id_regle, segment, parametre, seuil)
        resultats.append(metriques)
    
    # Convertir en DataFrame
    df_resultats = pd.DataFrame(resultats)
    
    # Tracer
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(df_resultats['seuil'], df_resultats['precision'], 'b-', label='Précision')
    ax.plot(df_resultats['seuil'], df_resultats['rappel'], 'g-', label='Rappel')
    ax.plot(df_resultats['seuil'], df_resultats['f1'], 'r-', label='Score F1')
    ax.plot(df_resultats['seuil'], df_resultats['specificite'], 'c-', label='Spécificité')
    
    # Trouver le seuil optimal basé sur le score F1
    idx_optimal = df_resultats['f1'].idxmax()
    seuil_optimal = df_resultats.loc[idx_optimal, 'seuil']
    
    # Ajouter une ligne verticale au seuil optimal
    ax.axvline(x=seuil_optimal, color='k', linestyle='--', 
               label=f'Seuil Optimal: {seuil_optimal}')
    
    ax.set_xlabel('Valeur du Seuil')
    ax.set_ylabel('Score')
    ax.set_title(f'Courbes de Performance des Seuils pour {id_regle}-{segment}-{parametre}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/courbes_seuil_{id_regle}_{segment}_{parametre}.png')
    plt.close()
    
    return df_resultats

# Extraire les combinaisons uniques règle-segment-paramètre
combinaisons_param = []
for regle in seuils:
    for segment in ['IND', 'CORP']:
        df_regle = seuils[regle]
        params = df_regle[df_regle['pop_group'] == segment]['parameter'].unique()
        for param in params:
            combinaisons_param.append((regle, segment, param))

print(f"\nTotal des combinaisons de paramètres à analyser: {len(combinaisons_param)}")

# Trouver les seuils optimaux pour tous les paramètres
seuils_optimaux = []

for regle, segment, param in combinaisons_param:
    print(f"Analyse de {regle}-{segment}-{param}...")
    
    # Trouver le seuil optimal
    seuil, metriques = trouver_seuil_optimal(regle, segment, param)
    
    if seuil is not None:
        # Tracer les courbes de seuil
        df_resultats = tracer_courbes_seuil(regle, segment, param)
        
        # Stocker les résultats
        seuils_optimaux.append({
            'regle': regle,
            'segment': segment,
            'parametre': param,
            'seuil_optimal': seuil,
            'precision': metriques['precision'],
            'rappel': metriques['rappel'],
            'f1': metriques['f1'],
            'specificite': metriques['specificite'],
            'vp': metriques['vp'],
            'fp': metriques['fp'],
            'vn': metriques['vn'],
            'fn': metriques['fn']
        })

# Convertir en DataFrame et sauvegarder
df_optimal = pd.DataFrame(seuils_optimaux)
df_optimal.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_optimaux.csv', index=False)

# Afficher les meilleurs paramètres par score F1
print("\nMeilleurs paramètres par score F1:")
print(df_optimal.sort_values('f1', ascending=False).head(10))

# Tracer la distribution des seuils optimaux
plt.figure(figsize=(12, 8))
sns.histplot(df_optimal['seuil_optimal'], bins=20, kde=True)
plt.title('Distribution des Seuils Optimaux')
plt.xlabel('Valeur du Seuil')
plt.ylabel('Fréquence')
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/distribution_seuils_optimaux.png')
plt.close()

# Tracer les scores F1 pour chaque paramètre
plt.figure(figsize=(15, 10))
meilleurs_params = df_optimal.sort_values('f1', ascending=False).head(20)
sns.barplot(x='f1', y='parametre', hue='regle', data=meilleurs_params)
plt.title('Scores F1 pour les 20 Meilleurs Paramètres')
plt.xlabel('Score F1')
plt.ylabel('Paramètre')
plt.tight_layout()
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/scores_f1_meilleurs_parametres.png')
plt.close()

# Fonction pour évaluer l'impact global des seuils optimisés
def evaluer_systeme_optimise():
    """
    Évaluer la performance du système avec des seuils optimisés
    """
    # Créer un dictionnaire des seuils optimaux
    dict_seuils = {}
    for _, ligne in df_optimal.iterrows():
        cle = (ligne['regle'], ligne['segment'], ligne['parametre'])
        dict_seuils[cle] = ligne['seuil_optimal']
    
    # Regrouper les données par alerte
    alertes = df.groupby('id_alerte')['is_issue'].max().reset_index()
    alertes['predit'] = 0
    
    # Appliquer les seuils optimisés
    for id_alerte in alertes['id_alerte']:
        donnees_alerte = df[df['id_alerte'] == id_alerte]
        
        # Vérifier si un paramètre dépasse son seuil
        for _, ligne in donnees_alerte.iterrows():
            cle = (ligne['rule_id'], ligne['segment'], ligne['param'])
            if cle in dict_seuils:
                if ligne['value'] >= dict_seuils[cle]:
                    alertes.loc[alertes['id_alerte'] == id_alerte, 'predit'] = 1
                    break
    
    # Calculer les métriques
    vp = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 1))
    fp = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 1))
    vn = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 0))
    fn = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 0))
    
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
    specificite = vn / (vn + fp) if (vn + fp) > 0 else 0
    
    # Afficher les résultats
    print("\nPerformance Globale du Système avec Seuils Optimisés:")
    print(f"Vrais Positifs: {vp}")
    print(f"Faux Positifs: {fp}")
    print(f"Vrais Négatifs: {vn}")
    print(f"Faux Négatifs: {fn}")
    print(f"Précision: {precision:.4f}")
    print(f"Rappel: {rappel:.4f}")
    print(f"Score F1: {f1:.4f}")
    print(f"Spécificité: {specificite:.4f}")
    
    # Créer la matrice de confusion
    mc = confusion_matrix(alertes['is_issue'], alertes['predit'])
    
    # Tracer la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non Alerté', 'Alerté'],
                yticklabels=['Pas un Problème', 'Problème'])
    plt.title('Matrice de Confusion pour le Système Optimisé')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/matrice_confusion_systeme_optimise.png')
    plt.close()
    
    return {
        'vp': vp, 'fp': fp, 'vn': vn, 'fn': fn,
        'precision': precision, 'rappel': rappel, 'f1': f1, 'specificite': specificite
    }

# Évaluer le système optimisé
metriques_optimisees = evaluer_systeme_optimise()

# Comparer avec la référence (seuils actuels)
def evaluer_systeme_reference():
    """
    Évaluer la performance du système avec les seuils actuels
    """
    # Regrouper les données par alerte
    alertes = df.groupby('id_alerte')['is_issue'].max().reset_index()
    alertes['predit'] = 0
    
    # Pour la référence, nous utilisons le score d'alerte existant
    scores_alerte = df.groupby('id_alerte')['alert_score'].max().reset_index()
    
    # Fusionner les scores d'alerte avec les alertes
    alertes = alertes.merge(scores_alerte, on='id_alerte', how='left')
    
    # Appliquer un seuil de 40 (seuil d'alerte standard)
    alertes['predit'] = (alertes['alert_score'] >= 40).astype(int)
    
    # Calculer les métriques
    vp = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 1))
    fp = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 1))
    vn = sum((alertes['is_issue'] == 0) & (alertes['predit'] == 0))
    fn = sum((alertes['is_issue'] == 1) & (alertes['predit'] == 0))
    
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
    specificite = vn / (vn + fp) if (vn + fp) > 0 else 0
    
    # Afficher les résultats
    print("\nPerformance du Système de Référence:")
    print(f"Vrais Positifs: {vp}")
    print(f"Faux Positifs: {fp}")
    print(f"Vrais Négatifs: {vn}")
    print(f"Faux Négatifs: {fn}")
    print(f"Précision: {precision:.4f}")
    print(f"Rappel: {rappel:.4f}")
    print(f"Score F1: {f1:.4f}")
    print(f"Spécificité: {specificite:.4f}")
    
    # Créer la matrice de confusion
    mc = confusion_matrix(alertes['is_issue'], alertes['predit'])
    
    # Tracer la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non Alerté', 'Alerté'],
                yticklabels=['Pas un Problème', 'Problème'])
    plt.title('Matrice de Confusion pour le Système de Référence')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/matrice_confusion_systeme_reference.png')
    plt.close()
    
    return {
        'vp': vp, 'fp': fp, 'vn': vn, 'fn': fn,
        'precision': precision, 'rappel': rappel, 'f1': f1, 'specificite': specificite
    }

# Évaluer le système de référence
metriques_reference = evaluer_systeme_reference()

# Comparer les métriques de référence et optimisées
comparaison = pd.DataFrame({
    'Métrique': ['Vrais Positifs', 'Faux Positifs', 'Vrais Négatifs', 'Faux Négatifs',
               'Précision', 'Rappel', 'Score F1', 'Spécificité'],
    'Référence': [metriques_reference['vp'], metriques_reference['fp'], 
                 metriques_reference['vn'], metriques_reference['fn'],
                 metriques_reference['precision'], metriques_reference['rappel'], 
                 metriques_reference['f1'], metriques_reference['specificite']],
    'Optimisé': [metriques_optimisees['vp'], metriques_optimisees['fp'], 
                  metriques_optimisees['vn'], metriques_optimisees['fn'],
                  metriques_optimisees['precision'], metriques_optimisees['rappel'], 
                  metriques_optimisees['f1'], metriques_optimisees['specificite']]
})

# Calculer l'amélioration
comparaison['Amélioration'] = comparaison['Optimisé'] - comparaison['Référence']
comparaison['Changement en %'] = (comparaison['Amélioration'] / comparaison['Référence']) * 100

# Formater les colonnes de pourcentage
for col in ['Précision', 'Rappel', 'Score F1', 'Spécificité']:
    idx = comparaison['Métrique'] == col
    comparaison.loc[idx, 'Référence'] = comparaison.loc[idx, 'Référence'].apply(lambda x: f"{x:.4f}")
    comparaison.loc[idx, 'Optimisé'] = comparaison.loc[idx, 'Optimisé'].apply(lambda x: f"{x:.4f}")
    comparaison.loc[idx, 'Amélioration'] = comparaison.loc[idx, 'Amélioration'].apply(lambda x: f"{x:.4f}")
    comparaison.loc[idx, 'Changement en %'] = comparaison.loc[idx, 'Changement en %'].apply(lambda x: f"{x:.2f}%")

print("\nComparaison des Systèmes de Référence et Optimisé:")
print(comparaison)

# Sauvegarder la comparaison en CSV
comparaison.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/comparaison_systemes.csv', index=False)

# Créer un graphique à barres comparant les métriques de référence et optimisées
metriques_a_tracer = ['Précision', 'Rappel', 'Score F1', 'Spécificité']
valeurs_reference = [float(comparaison.loc[comparaison['Métrique'] == m, 'Référence'].values[0]) for m in metriques_a_tracer]
valeurs_optimisees = [float(comparaison.loc[comparaison['Métrique'] == m, 'Optimisé'].values[0]) for m in metriques_a_tracer]

plt.figure(figsize=(12, 8))
x = np.arange(len(metriques_a_tracer))
largeur = 0.35

plt.bar(x - largeur/2, valeurs_reference, largeur, label='Référence')
plt.bar(x + largeur/2, valeurs_optimisees, largeur, label='Optimisé')

plt.xlabel('Métrique')
plt.ylabel('Score')
plt.title('Comparaison des Systèmes de Référence et Optimisé')
plt.xticks(x, metriques_a_tracer)
plt.legend()
plt.grid(True, axis='y')

# Ajouter les étiquettes de valeur
for i, v in enumerate(valeurs_reference):
    plt.text(i - largeur/2, v + 0.01, f"{v:.4f}", ha='center')
    
for i, v in enumerate(valeurs_optimisees):
    plt.text(i + largeur/2, v + 0.01, f"{v:.4f}", ha='center')

plt.tight_layout()
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/graphique_comparaison_systemes.png')
plt.close()

# Créer une fonction pour mettre à jour le fichier Excel des seuils avec les valeurs optimisées
def mettre_a_jour_seuils_excel():
    """
    Mettre à jour le fichier Excel des seuils avec les valeurs optimisées
    """
    # Créer une copie des seuils originaux
    seuils_mis_a_jour = {}
    for regle in seuils:
        seuils_mis_a_jour[regle] = seuils[regle].copy()
    
    # Mettre à jour avec les seuils optimisés
    for _, ligne in df_optimal.iterrows():
        regle = ligne['regle']
        segment = ligne['segment']
        parametre = ligne['parametre']
        seuil_optimal = ligne['seuil_optimal']
        
        # Trouver les lignes à mettre à jour
        masque = (seuils_mis_a_jour[regle]['pop_group'] == segment) & (seuils_mis_a_jour[regle]['parameter'] == parametre)
        
        # Mettre à jour les valeurs de seuil
        if any(masque):
            # Obtenir la valeur de seuil minimale pour ce paramètre
            seuil_min = seuils_mis_a_jour[regle].loc[masque, 'threshold_value'].min()
            
            # Calculer le facteur d'ajustement
            ajustement = seuil_optimal - seuil_min
            
            # Appliquer l'ajustement à toutes les valeurs de seuil pour ce paramètre
            seuils_mis_a_jour[regle].loc[masque, 'threshold_value'] += ajustement
    
    # Sauvegarder les seuils mis à jour dans un nouveau fichier Excel
    with pd.ExcelWriter('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_aml_optimises.xlsx') as writer:
        for regle in seuils_mis_a_jour:
            seuils_mis_a_jour[regle].to_excel(writer, sheet_name=regle, index=False)
    
    print("\nSeuils mis à jour sauvegardés dans seuils_aml_optimises.xlsx")

# Mettre à jour le fichier Excel des seuils
mettre_a_jour_seuils_excel()

print("\nAnalyse terminée!")