# Système de Détection d'Activités Suspectes pour ESM/SBC

Ce système permet de détecter les activités financières suspectes liées aux entreprises de services monétaires (ESM) et aux systèmes bancaires clandestins (SBC), conformément aux exigences du CANAFE.

## Composants

Le système comprend deux approches complémentaires de détection :

1. **Détection basée sur les règles** (`rule_based_detection.py`) :
   - Implémente des règles métier spécifiques pour détecter les comportements suspects
   - Analyse les transactions, virements et informations d'entité individuellement
   - Attribue des scores en fonction des règles déclenchées

2. **Détection basée sur l'analyse de réseau** (`network_analysis_detection.py`) :
   - Construit et analyse un réseau de transactions entre entités
   - Identifie les schémas complexes qui ne sont pas visibles au niveau individuel
   - Détecte les réseaux de mules financières et les systèmes bancaires clandestins

3. **Exécution combinée** (`run_combined_detection.py`) :
   - Intègre les deux approches pour une détection plus robuste
   - Génère des rapports détaillés des entités suspectes
   - Offre des visualisations de réseau pour l'analyse manuelle

## Utilisation

1. Assurez-vous que les données sont disponibles dans le répertoire `/Users/paulconerardy/Documents/AML/ESM/ESM:SBC/data` :
   - `synthetic_entity.csv` : Informations sur les entités
   - `synthetic_trx.csv` : Transactions
   - `synthetic_wires.csv` : Virements

2. Exécutez le script principal :
   ```bash
   python run_combined_detection.py

3. Suivez les instructions à l'écran pour choisir les options d'analyse.
4. Consultez les résultats dans les fichiers de sortie :
   
   - combined_detection_results.csv : Résultats pour toutes les entités
   - suspicious_entities_combined.csv : Détails des entités suspectes
   - Visualisations de réseau dans le dossier network_visualizations/
## Règles de détection
Le système implémente plusieurs règles pour détecter les activités suspectes, notamment :

- Virements importants suivis de transferts sortants
- Transactions avec des pays sanctionnés
- Dépôts en espèces fractionnés
- Utilisation de mules d'argent
- Transferts fréquents par courriel et virements internationaux
- Mélange de fonds entre comptes personnels et d'entreprise
- Volume inhabituellement élevé de dépôts
- Dépôts structurés sous le seuil de déclaration
- Retraits rapides après dépôts
- Virements importants provenant de sociétés de change
- Activité incompatible avec le profil ou le type d'entreprise
## Métriques d'analyse de réseau
L'analyse de réseau calcule plusieurs métriques pour identifier les comportements suspects :

- Centralité d'intermédiarité (betweenness centrality)
- Centralité de degré (degree centrality)
- Analyse des valeurs aberrantes dans les communautés
- Vitesse de transaction (transaction velocity)
- Schémas de transactions structurées
## Dépendances
- Python 3.7+
- pandas
- numpy
- networkx
- matplotlib (pour les visualisations)
## Performance
Le système est optimisé pour traiter de grands volumes de données :

- Filtrage des transactions par montant et période
- Limitation de la taille des réseaux analysés
- Calcul des métriques sur