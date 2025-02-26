import numpy as np
import pandas as pd

class StatisticalDetector:
    """
    Détecteur d'anomalies basé sur des méthodes statistiques.
    
    Cette classe implémente des méthodes statistiques classiques pour
    la détection d'anomalies, notamment:
    - Détection par score Z
    - Détection par écart interquartile (IQR)
    
    Attributs:
        threshold (float): Seuil pour la détection des anomalies
    """
    
    def __init__(self, threshold=3):
        self.threshold = threshold
    
    def zscore_detection(self, data):
        """
        Détecte les anomalies en utilisant la méthode du score Z.
        
        Cette méthode identifie les valeurs aberrantes en calculant
        leur distance à la moyenne en termes d'écarts-types.
        
        Args:
            data (np.array): Données à analyser
            
        Returns:
            np.array: Masque booléen des anomalies détectées
        """
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.threshold
    
    def iqr_detection(self, data):
        """
        Détecte les anomalies en utilisant la méthode de l'écart interquartile.
        
        Cette méthode identifie les valeurs aberrantes en utilisant
        les quartiles et l'écart interquartile pour définir des seuils.
        
        Args:
            data (np.array): Données à analyser
            
        Returns:
            np.array: Masque booléen des anomalies détectées
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)