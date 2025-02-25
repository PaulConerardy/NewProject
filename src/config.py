# Feature configurations
FEATURE_COLS = [
    # Amount-based features
    'amount_zscore_30d', 'amount_peer_dev', 
    'amount_mean_30d', 'amount_std_30d',
    'amount_roundness', 'near_threshold',
    
    # Frequency and velocity features
    'freq_hist_dev_30d', 'amount_velocity_30d',
    'tx_count_30d', 'time_since_last_tx',
    
    # Time-based features
    'outside_business_hours', 'is_weekend', 'is_night',
    
    # Risk features
    'overall_risk_score', 'risk_score', 'structuring_risk'
]

# Risk thresholds
RISK_THRESHOLDS = {
    'amount_zscore_30d': 3.0,
    'amount_peer_dev': 2.5,
    'freq_hist_dev_30d': 2.0,
    'structuring_risk': 0.8,
    'overall_risk_score': 0.7
}