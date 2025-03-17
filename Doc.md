### 1.2 Données et structure
Le système AML analysé comporte:

- 5 règles de détection: AML-TUA, AML-MNT, AML-CHG, AML-AUG, AML-FTF
- 2 segments de clientèle: particuliers (IND) et entreprises (CORP)
- Plusieurs paramètres par règle (ex: montant de transaction, fréquence, etc.)
- Un seuil de déclenchement d'alerte fixé à 40 points
Les données d'entrée contiennent:

- Des alertes historiques avec leurs scores
- L'indicateur de pertinence de l'alerte (is_issue)
- Les valeurs des paramètres ayant contribué à chaque alerte
## 2. Problème d'optimisation sous-jacent
### 2.1 Objectif
L'objectif est d'ajuster les seuils des paramètres pour:

- Réduire significativement le nombre de faux positifs
- Maintenir la détection des vrais positifs (cas problématiques réels)
### 2.2 Formalisation mathématique
Le problème peut être formalisé comme:

Maximiser: F(x) = 0.7 × (1 - FP'/FP) + 0.3 × (VP'/VP) - P

Où:

- x est un vecteur d'ajustements des seuils
- FP et FP' sont les faux positifs avant et après ajustement
- VP et VP' sont les vrais positifs avant et après ajustement
- P est une pénalité appliquée si VP' < VP
Sous contraintes:

- Les ajustements de seuil sont des valeurs entières
- Les ajustements sont limités à une plage [-5, 10]
### 2.3 Complexité du problème
Ce problème présente plusieurs défis:

- Espace de recherche combinatoire (pour n paramètres, 16^n combinaisons possibles)
- Fonction objectif non linéaire et non différentiable
- Compromis entre réduction des faux positifs et maintien des vrais positifs
Ces caractéristiques rendent les méthodes d'optimisation classiques peu adaptées et justifient l'utilisation d'un algorithme génétique.

## 3. Implémentation de l'algorithme génétique
### 3.1 Principes de l'algorithme génétique
L'algorithme génétique est une méthode d'optimisation inspirée de l'évolution naturelle qui:

- Manipule une population de solutions potentielles
- Évalue la qualité (fitness) de chaque solution
- Sélectionne les meilleures solutions pour reproduction
- Crée de nouvelles solutions par croisement et mutation
- Répète le processus sur plusieurs générations
### 3.2 Représentation des solutions
Chaque solution (individu) est représentée par:

- Un vecteur d'entiers de longueur égale au nombre de paramètres
- Chaque valeur représente l'ajustement à appliquer au seuil correspondant
- Les ajustements sont limités à l'intervalle [-5, 10]
### 3.3 Fonction d'évaluation (fitness)
La fonction d'évaluation calcule pour chaque solution:

1. Les nouveaux scores d'alerte après ajustement des seuils
2. Le nombre de vrais et faux positifs résultants
3. La rétention des vrais positifs (VP'/VP)
4. La réduction des faux positifs (1 - FP'/FP)
5. Une pénalité si des vrais positifs sont perdus
6. Un score final combinant ces métriques: 0.7 × réduction_FP + 0.3 × rétention_VP - pénalité
### 3.4 Opérateurs génétiques 3.4.1 Sélection
- Méthode de tournoi: k individus sont sélectionnés aléatoirement et le meilleur est retenu
- Favorise les solutions avec une meilleure fitness sans éliminer complètement la diversité 3.4.2 Croisement
- Croisement à deux points: échange de segments entre deux parents
- Permet de combiner les caractéristiques de bonnes solutions 3.4.3 Mutation
- Modification aléatoire de certains ajustements avec une probabilité donnée
- Maintient la diversité et permet d'explorer de nouvelles régions de l'espace de recherche
### 3.5 Élitisme et remplacement
- Conservation du meilleur individu à chaque génération
- Remplacement complet de la population par la descendance
### 3.6 Paramètres de l'algorithme
- Taille de population: 50 individus
- Nombre de générations: 30
- Probabilité de croisement: 0.5
- Probabilité de mutation: 0.2
## 4. Analyse des résultats
### 4.1 Évolution de la fitness
L'évolution de la fitness moyenne et maximale au cours des générations permet d'observer:

- La convergence progressive vers de meilleures solutions
- L'efficacité des opérateurs génétiques
- La diversité maintenue dans la population