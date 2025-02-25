import numpy as np
import pandas as pd

class StatisticalDetector:
    def __init__(self, threshold=3):
        self.threshold = threshold
    
    def zscore_detection(self, data):
        """
        Detect anomalies using Z-score method
        """
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.threshold
    
    def iqr_detection(self, data):
        """
        Detect anomalies using IQR method
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)