import pandas as pd
import numpy as np
import random

class ClientDataGenerator:
    def __init__(self, num_clients=1000, seed=None):
        """
        Initialise le générateur de données client.
        
        Paramètres:
        -----------
        num_clients : int
            Nombre d'enregistrements clients à générer
        seed : int, optionnel
            Graine aléatoire pour la reproductibilité
        """
        self.num_clients = num_clients
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Définir les valeurs possibles pour les variables catégorielles
        self.party_types = ['Individual', 'Business']
        self.mkt_groupes = ['Batisseur', 'Retraite', 'Jeunesse', 'Premium', 'Standard']
        self.naics_codes = ['45211', '52211', '62111', '72111', '44111', '54111']
        
    def generate_data(self):
        """
        Génère des données client synthétiques suivant la structure de data_filter.csv.
        
        Retourne:
        --------
        DataFrame
            DataFrame contenant les données client synthétiques
        """
        # Initialiser les listes pour stocker les données
        data = []
        
        for i in range(self.num_clients):
            # Déterminer le type de partie (50% Individuel, 50% Entreprise)
            party_type = np.random.choice(self.party_types)
            
            # Générer party_key (identifiant unique)
            party_key = random.randint(10000, 99999)
            
            # Générer le revenu en fonction du type de partie
            if party_type == 'Individual':
                income = int(np.random.lognormal(mean=10.5, sigma=0.5))  # Revenu individuel
                mkt_groupe = np.random.choice(self.mkt_groupes)  # Groupe de marché pour les individus
                naics_code = 'N/A'  # Pas de code NAICS pour les individus
                client_age = random.randint(18, 90)  # Âge entre 18 et 90
            else:  # Entreprise
                income = int(np.random.lognormal(mean=12, sigma=1))  # Revenu d'entreprise
                mkt_groupe = 'N/A'  # Pas de groupe de marché pour les entreprises
                naics_code = np.random.choice(self.naics_codes)  # Code NAICS pour les entreprises
                client_age = random.randint(1, 100)  # "Âge" de l'entreprise (années depuis la création)
            
            # Générer l'âge de la relation client (durée en tant que client)
            client_rel_age = round(random.uniform(0.1, 60.0), 1)
            
            # Générer le niveau de risque (1-5, 5 étant le risque le plus élevé)
            risk_level = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            
            # Générer des données transactionnelles (champs ACT_PROF)
            # Pour chaque type de transaction, générer le volume et la valeur
            transaction_types = ['009DJ', '001', '059DJ', 'RECEIVE_BP_INN', '003', '007']
            
            # Dictionnaire pour stocker les données de transaction
            txn_data = {}
            
            for txn_type in transaction_types:
                # Générer le volume (nombre de transactions)
                vol_key = f'ACT_PROF_{txn_type}_VOL'
                
                # Différents types de transactions ont des volumes typiques différents
                if txn_type in ['009DJ', '001']:
                    vol = int(np.random.lognormal(mean=8, sigma=1.2))
                elif txn_type == 'RECEIVE_BP_INN':
                    vol = int(np.random.lognormal(mean=5, sigma=1.5))
                else:
                    vol = int(np.random.lognormal(mean=4, sigma=2))
                
                txn_data[vol_key] = vol
                
                # Générer la valeur (montant total des transactions)
                val_key = f'ACT_PROF_{txn_type}_VAL'
                
                # La valeur est quelque peu corrélée avec le volume mais avec variation
                # Les clients à haut risque ont tendance à avoir des ratios valeur/volume plus élevés
                avg_txn_value = np.random.lognormal(mean=5 + (risk_level * 0.3), sigma=1)
                val = int(vol * avg_txn_value)
                
                txn_data[val_key] = val
            
            # Créer l'enregistrement client
            client_record = {
                'party_key': party_key,
                'party_type': party_type,
                'income': income,
                'mkt_groupe': mkt_groupe,
                'naics_code': naics_code,
                'client_rel_age': client_rel_age,
                'client_age': client_age,
                'risk_level': risk_level,
                **txn_data  # Ajouter tous les champs de données de transaction
            }
            
            data.append(client_record)
        
        # Convertir en DataFrame
        df = pd.DataFrame(data)
        
        # S'assurer que les colonnes sont dans le bon ordre
        column_order = [
            'party_key', 'party_type', 'income', 'mkt_groupe', 'naics_code', 
            'client_rel_age', 'client_age', 'risk_level',
            'ACT_PROF_009DJ_VOL', 'ACT_PROF_009DJ_VAL', 
            'ACT_PROF_001_VAL', 'ACT_PROF_001_VOL',
            'ACT_PROF_059DJ_VAL', 'ACT_PROF_059DJ_VOL',
            'ACT_PROF_RECEIVE_BP_INN_VAL', 'ACT_PROF_RECEIVE_BP_INN_VOL', 
            'ACT_PROF_003_VOL', 'ACT_PROF_003_VAL',
            'ACT_PROF_007_VOL', 'ACT_PROF_007_VAL'
        ]
        
        # Réorganiser les colonnes et retourner
        return df[column_order]
    
    def save_to_csv(self, filename='/Users/paulconerardy/Documents/Trae/ESM3/generated_client_data.csv'):
        """
        Génère les données client et les sauvegarde dans un fichier CSV.
        
        Paramètres:
        -----------
        filename : str
            Chemin pour sauvegarder les données générées
            
        Retourne:
        --------
        DataFrame
            Les données générées
        """
        data = self.generate_data()
        data.to_csv(filename, index=False)
        print(f"Généré {len(data)} enregistrements clients et sauvegardé dans {filename}")
        return data

# Exemple d'utilisation
if __name__ == "__main__":
    generator = ClientDataGenerator(num_clients=100000, seed=42)
    data = generator.save_to_csv()
    print("\nÉchantillon des données générées:")
    print(data.head())