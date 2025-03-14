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

# Créer un id_alerte pour regrouper par alerte
df['id_alerte'] = df['alert_date'] + '_' + df['account_number'].astype(str)

# Charger les seuils optimisés
seuils_optimaux = pd.read_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/seuils_optimaux.csv')

# Fonction pour évaluer la performance du système avec des seuils modifiés
def evaluer_systeme_avec_seuils_modifies(dict_seuils):
    """
    Évaluer la performance du système avec un ensemble de seuils modifiés
    """
    # Regrouper les données par alerte
    alertes = df.groupby('id_alerte')['is_issue'].max().reset_index()
    alertes['predit'] = 0
    
    # Appliquer les seuils
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
    
    return {
        'vp': vp, 'fp': fp, 'vn': vn, 'fn': fn,
        'precision': precision, 'rappel': rappel, 'f1': f1, 'specificite': specificite
    }

# Créer un dictionnaire des seuils optimaux
dict_seuils_optimaux = {}
for _, ligne in seuils_optimaux.iterrows():
    cle = (ligne['regle'], ligne['segment'], ligne['parametre'])
    dict_seuils_optimaux[cle] = ligne['seuil_optimal']

# Fonction pour effectuer une analyse de sensibilité sur un paramètre
def analyser_sensibilite_parametre(regle, segment, parametre, plage_pourcentage=0.5, nb_points=20):
    """
    Analyser comment les variations d'un seuil de paramètre affectent les métriques de performance
    
    Args:
        regle: ID de la règle
        segment: Segment de population
        parametre: Nom du paramètre
        plage_pourcentage: Plage de variation en pourcentage (0.5 = ±50%)
        nb_points: Nombre de points à évaluer
    """
    # Obtenir le seuil optimal
    cle = (regle, segment, parametre)
    if cle not in dict_seuils_optimaux:
        print(f"Pas de seuil optimal pour {regle}-{segment}-{parametre}")
        return None
    
    seuil_optimal = dict_seuils_optimaux[cle]
    
    # Créer une plage de valeurs de seuil à tester
    min_seuil = int(seuil_optimal * (1 - plage_pourcentage))
    max_seuil = int(seuil_optimal * (1 + plage_pourcentage))
    min_seuil = max(1, min_seuil)  # S'assurer que le seuil est au moins 1
    
    seuils_a_tester = np.linspace(min_seuil, max_seuil, nb_points)
    seuils_a_tester = [int(s) for s in seuils_a_tester]
    
    # Évaluer chaque seuil
    resultats = []
    for seuil in seuils_a_tester:
        # Créer une copie des seuils optimaux et modifier le seuil du paramètre cible
        dict_seuils_modifies = dict_seuils_optimaux.copy()
        dict_seuils_modifies[cle] = seuil
        
        # Évaluer la performance
        metriques = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
        
        # Stocker les résultats
        resultats.append({
            'seuil': seuil,
            'precision': metriques['precision'],
            'rappel': metriques['rappel'],
            'f1': metriques['f1'],
            'specificite': metriques['specificite'],
            'vp': metriques['vp'],
            'fp': metriques['fp'],
            'vn': metriques['vn'],
            'fn': metriques['fn']
        })
    
    # Convertir en DataFrame
    df_resultats = pd.DataFrame(resultats)
    
    # Tracer les résultats
    plt.figure(figsize=(12, 8))
    
    plt.plot(df_resultats['seuil'], df_resultats['precision'], 'b-', label='Précision')
    plt.plot(df_resultats['seuil'], df_resultats['rappel'], 'g-', label='Rappel')
    plt.plot(df_resultats['seuil'], df_resultats['f1'], 'r-', label='Score F1')
    plt.plot(df_resultats['seuil'], df_resultats['specificite'], 'c-', label='Spécificité')
    
    # Marquer le seuil optimal
    plt.axvline(x=seuil_optimal, color='k', linestyle='--', 
               label=f'Seuil Optimal: {seuil_optimal}')
    
    plt.xlabel('Valeur du Seuil')
    plt.ylabel('Score')
    plt.title(f'Analyse de Sensibilité pour {regle}-{segment}-{parametre}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/sensibilite_{regle}_{segment}_{parametre}.png')
    plt.close()
    
    return df_resultats

# Sélectionner les paramètres les plus importants pour l'analyse de sensibilité
top_parametres = seuils_optimaux.sort_values('f1', ascending=False).head(10)

print("Analyse de sensibilité pour les 10 paramètres les plus importants:")
for _, ligne in top_parametres.iterrows():
    regle = ligne['regle']
    segment = ligne['segment']
    parametre = ligne['parametre']
    
    print(f"Analyse de {regle}-{segment}-{parametre}...")
    df_resultats = analyser_sensibilite_parametre(regle, segment, parametre)

# Fonction pour effectuer une analyse de sensibilité globale
def analyser_sensibilite_globale(plage_pourcentage=0.2, nb_points=10):
    """
    Analyser comment les variations globales des seuils affectent les métriques de performance
    
    Args:
        plage_pourcentage: Plage de variation en pourcentage (0.2 = ±20%)
        nb_points: Nombre de points à évaluer
    """
    # Créer une plage de facteurs de variation
    facteurs = np.linspace(1 - plage_pourcentage, 1 + plage_pourcentage, nb_points)
    
    # Évaluer chaque facteur
    resultats = []
    for facteur in facteurs:
        # Créer des seuils modifiés en appliquant le facteur à tous les seuils
        dict_seuils_modifies = {}
        for cle, seuil in dict_seuils_optimaux.items():
            dict_seuils_modifies[cle] = max(1, int(seuil * facteur))
        
        # Évaluer la performance
        metriques = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
        
        # Stocker les résultats
        resultats.append({
            'facteur': facteur,
            'precision': metriques['precision'],
            'rappel': metriques['rappel'],
            'f1': metriques['f1'],
            'specificite': metriques['specificite'],
            'vp': metriques['vp'],
            'fp': metriques['fp'],
            'vn': metriques['vn'],
            'fn': metriques['fn']
        })
    
    # Convertir en DataFrame
    df_resultats = pd.DataFrame(resultats)
    
    # Tracer les résultats
    plt.figure(figsize=(12, 8))
    
    plt.plot(df_resultats['facteur'], df_resultats['precision'], 'b-', label='Précision')
    plt.plot(df_resultats['facteur'], df_resultats['rappel'], 'g-', label='Rappel')
    plt.plot(df_resultats['facteur'], df_resultats['f1'], 'r-', label='Score F1')
    plt.plot(df_resultats['facteur'], df_resultats['specificite'], 'c-', label='Spécificité')
    
    # Marquer le facteur optimal (1.0)
    plt.axvline(x=1.0, color='k', linestyle='--', 
               label='Facteur Optimal: 1.0')
    
    plt.xlabel('Facteur de Variation')
    plt.ylabel('Score')
    plt.title('Analyse de Sensibilité Globale')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/sensibilite_globale.png')
    plt.close()
    
    return df_resultats

print("\nAnalyse de sensibilité globale...")
df_resultats_global = analyser_sensibilite_globale()

# Fonction pour effectuer une analyse de sensibilité par règle
def analyser_sensibilite_par_regle(plage_pourcentage=0.2, nb_points=10):
    """
    Analyser comment les variations des seuils par règle affectent les métriques de performance
    
    Args:
        plage_pourcentage: Plage de variation en pourcentage (0.2 = ±20%)
        nb_points: Nombre de points à évaluer
    """
    # Obtenir les règles uniques
    regles = set([cle[0] for cle in dict_seuils_optimaux.keys()])
    
    # Créer une plage de facteurs de variation
    facteurs = np.linspace(1 - plage_pourcentage, 1 + plage_pourcentage, nb_points)
    
    # Pour chaque règle
    for regle in regles:
        # Évaluer chaque facteur
        resultats = []
        for facteur in facteurs:
            # Créer des seuils modifiés en appliquant le facteur aux seuils de cette règle
            dict_seuils_modifies = dict_seuils_optimaux.copy()
            for cle, seuil in dict_seuils_optimaux.items():
                if cle[0] == regle:
                    dict_seuils_modifies[cle] = max(1, int(seuil * facteur))
            
            # Évaluer la performance
            metriques = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
            
            # Stocker les résultats
            resultats.append({
                'facteur': facteur,
                'precision': metriques['precision'],
                'rappel': metriques['rappel'],
                'f1': metriques['f1'],
                'specificite': metriques['specificite'],
                'vp': metriques['vp'],
                'fp': metriques['fp'],
                'vn': metriques['vn'],
                'fn': metriques['fn']
            })
        
        # Convertir en DataFrame
        df_resultats = pd.DataFrame(resultats)
        
        # Tracer les résultats
        plt.figure(figsize=(12, 8))
        
        plt.plot(df_resultats['facteur'], df_resultats['precision'], 'b-', label='Précision')
        plt.plot(df_resultats['facteur'], df_resultats['rappel'], 'g-', label='Rappel')
        plt.plot(df_resultats['facteur'], df_resultats['f1'], 'r-', label='Score F1')
        plt.plot(df_resultats['facteur'], df_resultats['specificite'], 'c-', label='Spécificité')
        
        # Marquer le facteur optimal (1.0)
        plt.axvline(x=1.0, color='k', linestyle='--', 
                   label='Facteur Optimal: 1.0')
        
        plt.xlabel('Facteur de Variation')
        plt.ylabel('Score')
        plt.title(f'Analyse de Sensibilité pour la Règle {regle}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/sensibilite_regle_{regle}.png')
        plt.close()
    
    return None

print("\nAnalyse de sensibilité par règle...")
analyser_sensibilite_par_regle()

# Fonction pour effectuer une analyse de sensibilité par segment
def analyser_sensibilite_par_segment(plage_pourcentage=0.2, nb_points=10):
    """
    Analyser comment les variations des seuils par segment affectent les métriques de performance
    
    Args:
        plage_pourcentage: Plage de variation en pourcentage (0.2 = ±20%)
        nb_points: Nombre de points à évaluer
    """
    # Obtenir les segments uniques
    segments = set([cle[1] for cle in dict_seuils_optimaux.keys()])
    
    # Créer une plage de facteurs de variation
    facteurs = np.linspace(1 - plage_pourcentage, 1 + plage_pourcentage, nb_points)
    
    # Pour chaque segment
    for segment in segments:
        # Évaluer chaque facteur
        resultats = []
        for facteur in facteurs:
            # Créer des seuils modifiés en appliquant le facteur aux seuils de ce segment
            dict_seuils_modifies = dict_seuils_optimaux.copy()
            for cle, seuil in dict_seuils_optimaux.items():
                if cle[1] == segment:
                    dict_seuils_modifies[cle] = max(1, int(seuil * facteur))
            
            # Évaluer la performance
            metriques = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
            
            # Stocker les résultats
            resultats.append({
                'facteur': facteur,
                'precision': metriques['precision'],
                'rappel': metriques['rappel'],
                'f1': metriques['f1'],
                'specificite': metriques['specificite'],
                'vp': metriques['vp'],
                'fp': metriques['fp'],
                'vn': metriques['vn'],
                'fn': metriques['fn']
            })
        
        # Convertir en DataFrame
        df_resultats = pd.DataFrame(resultats)
        
        # Tracer les résultats
        plt.figure(figsize=(12, 8))
        
        plt.plot(df_resultats['facteur'], df_resultats['precision'], 'b-', label='Précision')
        plt.plot(df_resultats['facteur'], df_resultats['rappel'], 'g-', label='Rappel')
        plt.plot(df_resultats['facteur'], df_resultats['f1'], 'r-', label='Score F1')
        plt.plot(df_resultats['facteur'], df_resultats['specificite'], 'c-', label='Spécificité')
        
        # Marquer le facteur optimal (1.0)
        plt.axvline(x=1.0, color='k', linestyle='--', 
                   label='Facteur Optimal: 1.0')
        
        plt.xlabel('Facteur de Variation')
        plt.ylabel('Score')
        plt.title(f'Analyse de Sensibilité pour le Segment {segment}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/sensibilite_segment_{segment}.png')
        plt.close()
    
    return None

print("\nAnalyse de sensibilité par segment...")
analyser_sensibilite_par_segment()

# Créer un tableau récapitulatif des paramètres les plus sensibles
def identifier_parametres_sensibles():
    """
    Identifier les paramètres dont les variations ont le plus d'impact sur la performance
    """
    resultats = []
    
    # Pour chaque paramètre dans le top 10
    for _, ligne in top_parametres.iterrows():
        regle = ligne['regle']
        segment = ligne['segment']
        parametre = ligne['parametre']
        cle = (regle, segment, parametre)
        
        if cle not in dict_seuils_optimaux:
            continue
        
        seuil_optimal = dict_seuils_optimaux[cle]
        
        # Tester avec une augmentation de 20%
        seuil_augmente = max(1, int(seuil_optimal * 1.2))
        dict_seuils_modifies = dict_seuils_optimaux.copy()
        dict_seuils_modifies[cle] = seuil_augmente
        metriques_augmente = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
        
        # Tester avec une diminution de 20%
        seuil_diminue = max(1, int(seuil_optimal * 0.8))
        dict_seuils_modifies = dict_seuils_optimaux.copy()
        dict_seuils_modifies[cle] = seuil_diminue
        metriques_diminue = evaluer_systeme_avec_seuils_modifies(dict_seuils_modifies)
        
        # Calculer l'impact sur le score F1
        impact_augmente = metriques_augmente['f1'] - ligne['f1']
        impact_diminue = metriques_diminue['f1'] - ligne['f1']
        impact_absolu = max(abs(impact_augmente), abs(impact_diminue))
        
        # Stocker les résultats
        resultats.append({
            'regle': regle,
            'segment': segment,
            'parametre': parametre,
            'seuil_optimal': seuil_optimal,
            'impact_augmente': impact_augmente,
            'impact_diminue': impact_diminue,
            'impact_absolu': impact_absolu
        })
    
    # Convertir en DataFrame et trier par impact absolu
    df_resultats = pd.DataFrame(resultats)
    df_resultats = df_resultats.sort_values('impact_absolu', ascending=False)
    
    # Sauvegarder les résultats
    df_resultats.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/parametres_sensibles.csv', index=False)
    
    # Tracer un graphique à barres des paramètres les plus sensibles
    plt.figure(figsize=(15, 10))
    
    # Limiter aux 10 paramètres les plus sensibles
    df_plot = df_resultats.head(10)
    
    # Créer des étiquettes combinées pour l'axe y
    etiquettes = [f"{r}-{s}-{p}" for r, s, p in zip(df_plot['regle'], df_plot['segment'], df_plot['parametre'])]
    
    # Tracer les barres pour l'impact de l'augmentation et de la diminution
    plt.barh(etiquettes, df_plot['impact_augmente'], color='green', alpha=0.7, label='+20%')
    plt.barh(etiquettes, df_plot['impact_diminue'], color='red', alpha=0.7, label='-20%')
    
    plt.xlabel('Impact sur le Score F1')
    plt.ylabel('Paramètre')
    plt.title('Impact des Variations de Seuil sur le Score F1')
    plt.legend()
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/versions_francaises/impact_parametres_sensibles.png')
    plt.close()
    
    return df_resultats

print("\nIdentification des paramètres les plus sensibles...")
df_parametres_sensibles = identifier_parametres_sensibles()
print("\nParamètres les plus sensibles:")
print(df_parametres_sensibles.head(5))

print("\nAnalyse de sensibilité terminée!")