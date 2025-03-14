import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Charger les données
chemin_donnees = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df = pd.read_csv(chemin_donnees)

# Charger les seuils
chemin_excel = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
seuils = {}
for regle in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    seuils[regle] = pd.read_excel(chemin_excel, sheet_name=regle)

# Regrouper les données par alerte (date_alerte + numero_compte)
df['id_alerte'] = df['alert_date'] + '_' + df['account_number'].astype(str)
alertes = df.groupby('id_alerte')['is_issue'].max().reset_index()
donnees_alertes = df.groupby('id_alerte')

# Extraire les paramètres uniques pour chaque règle et segment
cles_param = []
for regle in seuils:
    for segment in ['IND', 'CORP']:
        df_regle = seuils[regle]
        params = df_regle[df_regle['pop_group'] == segment]['parameter'].unique()
        for param in params:
            cles_param.append((regle, segment, param))

# Créer un mappage des valeurs de paramètres aux scores basé sur les seuils
def obtenir_score_pour_valeur(regle, segment, param, valeur, ajustements_seuil):
    df_regle = seuils[regle]
    df_param = df_regle[(df_regle['pop_group'] == segment) & (df_regle['parameter'] == param)]
    
    # Trier par valeur de seuil
    df_param = df_param.sort_values('threshold_value')
    
    # Appliquer les ajustements aux seuils
    cle = (regle, segment, param)
    if cle in ajustements_seuil:
        ajustement = ajustements_seuil[cle]
        # Ajuster les seuils avec des ajustements entiers
        df_param['threshold_value'] = df_param['threshold_value'] + ajustement
    
    # Trouver le score le plus élevé où valeur >= seuil
    score = 0
    for _, ligne in df_param.iterrows():
        if valeur >= ligne['threshold_value']:
            score = ligne['score']
        else:
            break
    
    return score

# Évaluer un ensemble d'ajustements de seuil
def evaluer_individu(individu):
    # Convertir l'individu en dictionnaire d'ajustements
    ajustements_seuil = {}
    for i, cle in enumerate(cles_param):
        ajustements_seuil[cle] = individu[i]
    
    # Calculer les nouveaux scores pour chaque alerte
    vrais_positifs = 0
    faux_positifs = 0
    vrais_positifs_avant = 0
    faux_positifs_avant = 0
    
    for id_alerte, groupe in donnees_alertes:
        est_probleme = groupe['is_issue'].iloc[0]
        score_original = groupe['alert_score'].iloc[0]
        
        # Compter les alertes originales
        if score_original >= 40:
            if est_probleme == 1:
                vrais_positifs_avant += 1
            else:
                faux_positifs_avant += 1
        
        # Calculer le nouveau score avec les seuils ajustés
        nouveau_score = 0
        for _, ligne in groupe.iterrows():
            regle = ligne['rule_id']
            segment = ligne['segment']
            param = ligne['param']
            valeur = ligne['value']
            
            # Obtenir le nouveau score basé sur les seuils ajustés
            score_param = obtenir_score_pour_valeur(regle, segment, param, valeur, ajustements_seuil)
            nouveau_score += score_param
        
        # Compter les alertes avec les nouveaux seuils
        if nouveau_score >= 40:
            if est_probleme == 1:
                vrais_positifs += 1
            else:
                faux_positifs += 1
    
    # Calculer la fitness: maximiser la rétention de VP et minimiser les FP
    retention_vp = vrais_positifs / max(1, vrais_positifs_avant)
    reduction_fp = 1 - (faux_positifs / max(1, faux_positifs_avant))
    
    # Pénaliser fortement si nous perdons des vrais positifs
    if vrais_positifs < vrais_positifs_avant:
        penalite_vp = 0.5 * (vrais_positifs_avant - vrais_positifs) / vrais_positifs_avant
    else:
        penalite_vp = 0
    
    # Fitness finale: équilibre entre réduction de FP et rétention de VP
    fitness = (0.7 * reduction_fp + 0.3 * retention_vp) - penalite_vp
    
    return fitness

# Implémentation personnalisée des composants de l'algorithme génétique

# Créer un individu aléatoire (liste d'ajustements entiers)
def creer_individu():
    return [random.randint(-5, 10) for _ in range(len(cles_param))]

# Créer la population initiale
def creer_population(taille_pop):
    return [creer_individu() for _ in range(taille_pop)]

# Sélection par tournoi
def selection(population, fitness_values, k=3):
    # Sélectionner k individus aléatoires et retourner le meilleur
    indices_selectionnes = random.sample(range(len(population)), k)
    meilleur_indice = max(indices_selectionnes, key=lambda i: fitness_values[i])
    return population[meilleur_indice]

# Croisement à deux points
def croisement(parent1, parent2):
    # Choisir deux points aléatoires
    longueur = len(parent1)
    if longueur < 2:
        return parent1[:], parent2[:]
    
    point1 = random.randint(1, longueur-1)
    point2 = random.randint(1, longueur-1)
    
    # S'assurer que point1 <= point2
    if point1 > point2:
        point1, point2 = point2, point1
    
    # Créer des enfants en échangeant des segments
    enfant1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    enfant2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return enfant1, enfant2

# Mutation
def muter(individu, taux_mutation=0.2, min=-5, max=10):
    for i in range(len(individu)):
        if random.random() < taux_mutation:
            individu[i] = random.randint(min, max)
    return individu

# Évaluer l'impact des ajustements de seuil
def evaluer_impact(ajustements_seuil):
    alertes_avant = {'VP': 0, 'FP': 0}
    alertes_apres = {'VP': 0, 'FP': 0}
    
    # Traiter chaque alerte
    for id_alerte, groupe in donnees_alertes:
        est_probleme = groupe['is_issue'].iloc[0]
        score_original = groupe['alert_score'].iloc[0]
        
        # Compter les alertes originales
        if score_original >= 40:
            if est_probleme == 1:
                alertes_avant['VP'] += 1
            else:
                alertes_avant['FP'] += 1
        
        # Calculer le nouveau score avec les seuils ajustés
        nouveau_score = 0
        for _, ligne in groupe.iterrows():
            regle = ligne['rule_id']
            segment = ligne['segment']
            param = ligne['param']
            valeur = ligne['value']
            
            # Obtenir le nouveau score basé sur les seuils ajustés
            score_param = obtenir_score_pour_valeur(regle, segment, param, valeur, ajustements_seuil)
            nouveau_score += score_param
        
        # Compter les alertes avec les nouveaux seuils
        if nouveau_score >= 40:
            if est_probleme == 1:
                alertes_apres['VP'] += 1
            else:
                alertes_apres['FP'] += 1
    
    # Afficher les résultats
    print("\nImpact des ajustements de seuil:")
    print(f"Avant: {alertes_avant['VP']} vrais positifs, {alertes_avant['FP']} faux positifs")
    print(f"Après: {alertes_apres['VP']} vrais positifs, {alertes_apres['FP']} faux positifs")
    
    if alertes_avant['FP'] > 0:
        reduction_fp = (alertes_avant['FP'] - alertes_apres['FP']) / alertes_avant['FP'] * 100
        print(f"Réduction des faux positifs: {reduction_fp:.2f}%")
    
    if alertes_avant['VP'] > 0:
        retention_vp = alertes_apres['VP'] / alertes_avant['VP'] * 100
        print(f"Rétention des vrais positifs: {retention_vp:.2f}%")
    
    return alertes_avant, alertes_apres

# Fonction principale de l'algorithme génétique
def algorithme_genetique(taille_pop=50, generations=30, prob_croisement=0.5, prob_mutation=0.2):
    # Définir la graine aléatoire pour la reproductibilité
    random.seed(42)
    
    # Créer la population initiale
    population = creer_population(taille_pop)
    
    # Suivre le meilleur individu et l'historique de fitness
    meilleur_individu = None
    meilleure_fitness = -float('inf')
    historique_fitness_moy = []
    historique_fitness_max = []
    
    # Évaluer la population initiale
    fitness_values = [evaluer_individu(ind) for ind in population]
    
    # Boucle principale d'évolution
    for gen in tqdm(range(generations), desc="Évolution en cours"):
        # Sélectionner les parents et créer la descendance
        descendance = []
        
        # Élitisme: garder le meilleur individu
        indice_elite = fitness_values.index(max(fitness_values))
        elite = population[indice_elite]
        descendance.append(elite)
        
        # Créer le reste de la descendance par sélection, croisement et mutation
        while len(descendance) < taille_pop:
            # Sélectionner les parents
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            
            # Appliquer le croisement avec probabilité
            if random.random() < prob_croisement:
                enfant1, enfant2 = croisement(parent1, parent2)
            else:
                enfant1, enfant2 = parent1[:], parent2[:]
            
            # Appliquer la mutation avec probabilité
            enfant1 = muter(enfant1, prob_mutation)
            enfant2 = muter(enfant2, prob_mutation)
            
            # Ajouter à la descendance
            descendance.append(enfant1)
            if len(descendance) < taille_pop:
                descendance.append(enfant2)
        
        # Remplacer la population par la descendance
        population = descendance
        
        # Évaluer la nouvelle population
        fitness_values = [evaluer_individu(ind) for ind in population]
        
        # Suivre les statistiques
        gen_moy = sum(fitness_values) / len(fitness_values)
        gen_max = max(fitness_values)
        historique_fitness_moy.append(gen_moy)
        historique_fitness_max.append(gen_max)
        
        # Mettre à jour le meilleur individu si nécessaire
        if gen_max > meilleure_fitness:
            meilleure_fitness = gen_max
            indice_meilleur = fitness_values.index(gen_max)
            meilleur_individu = population[indice_meilleur]
        
        # Afficher la progression
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"Génération {gen+1}: Fitness moyenne = {gen_moy:.4f}, Fitness max = {gen_max:.4f}")
    
    # Convertir le meilleur individu en ajustements de seuil
    meilleurs_ajustements = {}
    for i, cle in enumerate(cles_param):
        meilleurs_ajustements[cle] = meilleur_individu[i]
    
    # Sauvegarder les meilleurs ajustements
    df_ajustements = pd.DataFrame([
        {'regle': cle[0], 'segment': cle[1], 'parametre': cle[2], 'ajustement': valeur}
        for cle, valeur in meilleurs_ajustements.items()
    ])
    
    # Trier par valeur d'ajustement pour voir quels paramètres ont été ajustés le plus
    df_ajustements = df_ajustements.sort_values('ajustement', ascending=False)
    df_ajustements.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/ajustements_optimises_sans_deap.csv', index=False)
    
    # Évaluer l'impact de la meilleure solution
    evaluer_impact(meilleurs_ajustements)
    
    # Tracer l'évolution de la fitness
    plt.figure(figsize=(10, 6))
    plage_gen = range(generations)
    plt.plot(plage_gen, historique_fitness_moy, 'k-', label='Fitness Moyenne')
    plt.plot(plage_gen, historique_fitness_max, 'r-', label='Meilleure Fitness')
    plt.title('Évolution de la Fitness')
    plt.xlabel('Génération')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/evolution_fitness_sans_deap.png')
    plt.close()
    
    return meilleur_individu, meilleurs_ajustements, historique_fitness_moy, historique_fitness_max

# Fonction pour mettre à jour le fichier Excel des seuils avec les valeurs optimisées
def mettre_a_jour_seuils_excel(meilleurs_ajustements):
    # Créer une copie des seuils originaux
    seuils_mis_a_jour = {}
    for regle in seuils:
        seuils_mis_a_jour[regle] = seuils[regle].copy()
    
    # Mettre à jour avec les seuils optimisés
    for cle, ajustement in meilleurs_ajustements.items():
        regle, segment, parametre = cle
        
        # Trouver les lignes à mettre à jour
        masque = (seuils_mis_a_jour[regle]['pop_group'] == segment) & (seuils_mis_a_jour[regle]['parameter'] == parametre)
        
        # Mettre à jour les valeurs de seuil
        if any(masque):
            seuils_mis_a_jour[regle].loc[masque, 'threshold_value'] += ajustement
    
    # Sauvegarder les seuils mis à jour dans un nouveau fichier Excel
    with pd.ExcelWriter('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_aml_optimises_sans_deap.xlsx') as writer:
        for regle in seuils_mis_a_jour:
            seuils_mis_a_jour[regle].to_excel(writer, sheet_name=regle, index=False)
    
    print("\nSeuils mis à jour sauvegardés dans seuils_aml_optimises_sans_deap.xlsx")

if __name__ == "__main__":
    print("Démarrage de l'optimisation par algorithme génétique...")
    print(f"Nombre de paramètres à optimiser: {len(cles_param)}")
    
    # Exécuter l'algorithme génétique
    meilleur_individu, meilleurs_ajustements, historique_fitness_moy, historique_fitness_max = algorithme_genetique(
        taille_pop=50, 
        generations=30, 
        prob_croisement=0.5, 
        prob_mutation=0.2
    )
    
    # Mettre à jour le fichier Excel des seuils
    mettre_a_jour_seuils_excel(meilleurs_ajustements)
    
    print("Optimisation terminée!")