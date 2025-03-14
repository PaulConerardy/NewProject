import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Définir le style des graphiques
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Charger les données
chemin_donnees = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df = pd.read_csv(chemin_donnees)

# Charger les seuils
chemin_excel = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
seuils = {}
for regle in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    seuils[regle] = pd.read_excel(chemin_excel, sheet_name=regle)

# Créer un id_alerte pour regrouper par alerte
df['id_alerte'] = df['alert_date'] + '_' + df['account_number'].astype(str)

# Extraire les paramètres uniques pour chaque règle et segment
cles_param = []
for regle in seuils:
    for segment in ['IND', 'CORP']:
        df_regle = seuils[regle]
        params = df_regle[df_regle['pop_group'] == segment]['parameter'].unique()
        for param in params:
            cles_param.append((regle, segment, param))

print(f"Nombre de paramètres à optimiser: {len(cles_param)}")

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

# Fonction pour trouver le seuil optimal pour un paramètre en utilisant une approche statistique
def trouver_seuil_optimal_statistique(id_regle, segment, parametre):
    """
    Trouver la valeur de seuil optimale pour un paramètre en utilisant une approche statistique
    """
    # Filtrer les données pour cette règle-segment-paramètre
    donnees_param = df[(df['rule_id'] == id_regle) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parametre)]
    
    if donnees_param.empty:
        print(f"Pas de données pour {id_regle}-{segment}-{parametre}")
        return None, None
    
    # Séparer les valeurs pour les problèmes et non-problèmes
    valeurs_probleme = donnees_param[donnees_param['is_issue'] == 1]['value']
    valeurs_non_probleme = donnees_param[donnees_param['is_issue'] == 0]['value']
    
    if valeurs_probleme.empty or valeurs_non_probleme.empty:
        print(f"Données insuffisantes pour {id_regle}-{segment}-{parametre}")
        return None, None
    
    # Calculer les statistiques
    moy_probleme = valeurs_probleme.mean()
    moy_non_probleme = valeurs_non_probleme.mean()
    ecart_type_probleme = valeurs_probleme.std()
    ecart_type_non_probleme = valeurs_non_probleme.std()
    
    # Si les écarts-types sont trop petits, utiliser une valeur minimale
    ecart_type_probleme = max(ecart_type_probleme, 1e-6)
    ecart_type_non_probleme = max(ecart_type_non_probleme, 1e-6)
    
    # Calculer le seuil optimal en utilisant la formule de Fisher
    # Cette formule trouve le point où les deux distributions normales se croisent
    if moy_probleme > moy_non_probleme:
        # Cas normal: les problèmes ont des valeurs plus élevées
        a = 1/(2*ecart_type_probleme**2) - 1/(2*ecart_type_non_probleme**2)
        b = moy_non_probleme/(ecart_type_non_probleme**2) - moy_probleme/(ecart_type_probleme**2)
        c = moy_probleme**2/(2*ecart_type_probleme**2) - moy_non_probleme**2/(2*ecart_type_non_probleme**2) - np.log(ecart_type_probleme/ecart_type_non_probleme)
        
        # Résoudre l'équation quadratique
        if a == 0:
            # Si a est zéro, l'équation est linéaire
            seuil = -c/b if b != 0 else moy_probleme
        else:
            # Sinon, utiliser la formule quadratique
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                # Pas de solution réelle, utiliser la moyenne des moyennes
                seuil = (moy_probleme + moy_non_probleme) / 2
            else:
                # Choisir la solution qui est entre les deux moyennes
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                if moy_non_probleme <= x1 <= moy_probleme or moy_probleme <= x1 <= moy_non_probleme:
                    seuil = x1
                elif moy_non_probleme <= x2 <= moy_probleme or moy_probleme <= x2 <= moy_non_probleme:
                    seuil = x2
                else:
                    # Si aucune solution n'est entre les moyennes, utiliser la moyenne des moyennes
                    seuil = (moy_probleme + moy_non_probleme) / 2
    else:
        # Cas inverse: les non-problèmes ont des valeurs plus élevées
        # Dans ce cas, nous voulons un seuil inférieur pour capturer les problèmes
        seuil = moy_probleme
    
    # S'assurer que le seuil est un nombre entier positif
    seuil = max(1, int(round(seuil)))
    
    # Évaluer la performance du seuil
    metriques = evaluer_seuil(id_regle, segment, parametre, seuil)
    
    # Tracer la distribution des valeurs
    plt.figure(figsize=(12, 8))
    sns.histplot(valeurs_non_probleme, color='blue', alpha=0.5, label='Non-Problèmes', kde=True)
    sns.histplot(valeurs_probleme, color='red', alpha=0.5, label='Problèmes', kde=True)
    plt.axvline(x=seuil, color='green', linestyle='--', label=f'Seuil Optimal: {seuil}')
    plt.title(f'Distribution des Valeurs pour {id_regle}-{segment}-{parametre}')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/distribution_{id_regle}_{segment}_{parametre}.png')
    plt.close()
    
    return seuil, metriques

# Trouver les seuils optimaux pour tous les paramètres
seuils_optimaux = []

for regle, segment, param in tqdm(cles_param, desc="Optimisation des seuils"):
    # Trouver le seuil optimal
    seuil, metriques = trouver_seuil_optimal_statistique(regle, segment, param)
    
    if seuil is not None:
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
df_optimal.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_optimaux_statistiques.csv', index=False)

# Afficher les meilleurs paramètres par score F1
print("\nMeilleurs paramètres par score F1:")
print(df_optimal.sort_values('f1', ascending=False).head(10))

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
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/matrice_confusion_systeme_optimise_stat.png')
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
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/matrice_confusion_systeme_reference_stat.png')
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
comparaison.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/comparaison_systemes_statistique.csv', index=False)

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
plt.title('Comparaison des Systèmes de Référence et Optimisé (Approche Statistique)')
plt.xticks(x, metriques_a_tracer)
plt.legend()
plt.grid(True, axis='y')

# Ajouter les étiquettes de valeur
for i, v in enumerate(valeurs_reference):
    plt.text(i - largeur/2, v + 0.01, f"{v:.4f}", ha='center')
    
for i, v in enumerate(valeurs_optimisees):
    plt.text(i + largeur/2, v + 0.01, f"{v:.4f}", ha='center')

plt.tight_layout()
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/graphique_comparaison_systemes_statistique.png')
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
    with pd.ExcelWriter('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_aml_optimises_statistique.xlsx') as writer:
        for regle in seuils_mis_a_jour:
            seuils_mis_a_jour[regle].to_excel(writer, sheet_name=regle, index=False)
    
    print("\nSeuils mis à jour sauvegardés dans seuils_aml_optimises_statistique.xlsx")

# Mettre à jour le fichier Excel des seuils
mettre_a_jour_seuils_excel()

print("\nAnalyse statistique terminée!")