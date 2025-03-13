import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Load the data
data_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df = pd.read_csv(data_path)

# Load the thresholds
excel_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
thresholds = {}
for rule in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    thresholds[rule] = pd.read_excel(excel_path, sheet_name=rule)

# Group data by alert (alert_date + account_number)
df['alert_id'] = df['alert_date'] + '_' + df['account_number'].astype(str)
alerts_data = df.groupby('alert_id')

# Extract unique rule-segment-parameter combinations
param_combinations = []
for rule in thresholds:
    for segment in ['IND', 'CORP']:
        rule_df = thresholds[rule]
        params = rule_df[rule_df['pop_group'] == segment]['parameter'].unique()
        for param in params:
            param_combinations.append((rule, segment, param))

# Function to evaluate the impact of a threshold adjustment
def evaluate_threshold_adjustment(rule, segment, param, percentile_value):
    # Filter data for this rule-segment-parameter
    param_data = df[(df['rule_id'] == rule) & (df['segment'] == segment) & (df['param'] == param)]
    
    # If no data for this combination, return default metrics
    if param_data.empty:
        return {
            'tp_before': 0, 'fp_before': 0, 
            'tp_after': 0, 'fp_after': 0,
            'tp_retention': 1.0, 'fp_reduction': 0.0
        }
    
    # Get the original threshold and score mapping
    rule_df = thresholds[rule]
    threshold_df = rule_df[(rule_df['pop_group'] == segment) & (rule_df['parameter'] == param)]
    
    # Count original alerts
    tp_before = 0
    fp_before = 0
    tp_after = 0
    fp_after = 0
    
    # Group by alert_id to avoid double counting
    for alert_id, group in param_data.groupby('alert_id'):
        is_issue = group['is_issue'].iloc[0]
        original_score = group['score'].iloc[0]
        value = group['value'].iloc[0]
        
        # Count before adjustment
        if original_score > 0:
            if is_issue == 1:
                tp_before += 1
            else:
                fp_before += 1
        
        # Apply new threshold
        new_score = 0
        if value >= percentile_value:
            # Find the appropriate score from the threshold table
            for _, row in threshold_df.sort_values('threshold_value').iterrows():
                if value >= row['threshold_value']:
                    new_score = row['score']
                else:
                    break
        
        # Count after adjustment
        if new_score > 0:
            if is_issue == 1:
                tp_after += 1
            else:
                fp_after += 1
    
    # Calculate metrics
    tp_retention = tp_after / max(1, tp_before) if tp_before > 0 else 1.0
    fp_reduction = 1 - (fp_after / max(1, fp_before)) if fp_before > 0 else 0.0
    
    return {
        'tp_before': tp_before, 'fp_before': fp_before, 
        'tp_after': tp_after, 'fp_after': fp_after,
        'tp_retention': tp_retention, 'fp_reduction': fp_reduction
    }

# Function to find optimal threshold for a parameter
def find_optimal_threshold(rule, segment, param):
    # Filter data for this rule-segment-parameter
    param_data = df[(df['rule_id'] == rule) & (df['segment'] == segment) & (df['param'] == param)]
    
    if param_data.empty:
        print(f"No data for {rule}-{segment}-{param}")
        return None, None
    
    # Get unique integer values from the data instead of percentiles
    # Sort them to ensure we test thresholds in ascending order
    unique_values = sorted(set(int(round(val)) for val in param_data['value']))
    
    # If no unique values, return None
    if not unique_values:
        return None, None
    
    # Evaluate each integer value as a potential threshold
    results = []
    for int_value in unique_values:
        metrics = evaluate_threshold_adjustment(rule, segment, param, int_value)
        
        # Calculate a combined score (balance between TP retention and FP reduction)
        # Prioritize TP retention more (0.7 weight)
        combined_score = 0.3 * metrics['fp_reduction'] + 0.7 * metrics['tp_retention']
        
        results.append({
            'percentile_value': int_value,  # Using integer values directly
            'tp_retention': metrics['tp_retention'],
            'fp_reduction': metrics['fp_reduction'],
            'combined_score': combined_score
        })
    
    # Find the optimal threshold (highest combined score)
    results_df = pd.DataFrame(results)
    
    # If no results, return None
    if results_df.empty:
        return None, None
        
    optimal_idx = results_df['combined_score'].idxmax()
    optimal_value = results_df.loc[optimal_idx, 'percentile_value']
    optimal_score = results_df.loc[optimal_idx, 'combined_score']
    
    # No need to round since we're already using integers
    return int(optimal_value), optimal_score

# Find optimal thresholds for all parameters
optimal_thresholds = {}
for rule, segment, param in param_combinations:
    print(f"Optimizing {rule}-{segment}-{param}...")
    optimal_value, optimal_score = find_optimal_threshold(rule, segment, param)
    
    if optimal_value is not None:
        optimal_thresholds[(rule, segment, param)] = {
            'optimal_value': optimal_value,
            'score': optimal_score
        }

# Save the optimized thresholds
results = []
for (rule, segment, param), data in optimal_thresholds.items():
    results.append({
        'rule': rule,
        'segment': segment,
        'parameter': param,
        'optimal_threshold': data['optimal_value'],
        'score': data['score']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/optimized_thresholds_statistical.csv', index=False)

# Evaluate the overall impact of the optimized thresholds
def evaluate_overall_impact():
    # Original alerts
    alerts_before = {'TP': 0, 'FP': 0}
    alerts_after = {'TP': 0, 'FP': 0}
    
    # Process each alert
    for alert_id, group in alerts_data:
        is_issue = group['is_issue'].iloc[0]
        original_score = group['alert_score'].iloc[0]
        
        # Count original alerts
        if original_score >= 40:
            if is_issue == 1:
                alerts_before['TP'] += 1
            else:
                alerts_before['FP'] += 1
        
        # Calculate new score with optimized thresholds
        new_score = 0
        for _, row in group.iterrows():
            rule = row['rule_id']
            segment = row['segment']
            param = row['param']
            value = row['value']
            
            # Get score based on optimized threshold
            key = (rule, segment, param)
            if key in optimal_thresholds:
                optimal_value = optimal_thresholds[key]['optimal_value']
                
                # Find the appropriate score from the threshold table
                rule_df = thresholds[rule]
                threshold_df = rule_df[(rule_df['pop_group'] == segment) & (rule_df['parameter'] == param)]
                
                param_score = 0
                if value >= optimal_value:
                    for _, t_row in threshold_df.sort_values('threshold_value').iterrows():
                        if value >= t_row['threshold_value']:
                            param_score = t_row['score']
                        else:
                            break
                
                new_score += param_score
        
        # Count alerts with new thresholds
        if new_score >= 40:
            if is_issue == 1:
                alerts_after['TP'] += 1
            else:
                alerts_after['FP'] += 1
    
    # Print results
    print("\nOverall Impact of Optimized Thresholds:")
    print(f"Before: {alerts_before['TP']} true positives, {alerts_before['FP']} false positives")
    print(f"After:  {alerts_after['TP']} true positives, {alerts_after['FP']} false positives")
    
    if alerts_before['FP'] > 0:
        fp_reduction = (alerts_before['FP'] - alerts_after['FP']) / alerts_before['FP'] * 100
        print(f"False Positive Reduction: {fp_reduction:.2f}%")
    
    if alerts_before['TP'] > 0:
        tp_retention = alerts_after['TP'] / alerts_before['TP'] * 100
        print(f"True Positive Retention: {tp_retention:.2f}%")
    
    return alerts_before, alerts_after

# Visualize the results
def visualize_results(alerts_before, alerts_after):
    # Create a bar chart showing before/after comparison
    labels = ['True Positives', 'False Positives']
    before_values = [alerts_before['TP'], alerts_before['FP']]
    after_values = [alerts_after['TP'], alerts_after['FP']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before_values, width, label='Before Optimization')
    rects2 = ax.bar(x + width/2, after_values, width, label='After Optimization')
    
    ax.set_ylabel('Number of Alerts')
    ax.set_title('Impact of Statistical Threshold Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/statistical_results_comparison.png')
    
    # Create a histogram of threshold adjustments
    plt.figure(figsize=(12, 8))
    
    # Get top parameters by impact
    param_impact = []
    for (rule, segment, param), data in optimal_thresholds.items():
        # Get the original threshold
        rule_df = thresholds[rule]
        threshold_df = rule_df[(rule_df['pop_group'] == segment) & (rule_df['parameter'] == param)]
        if not threshold_df.empty:
            original_threshold = threshold_df['threshold_value'].min()
            param_impact.append({
                'param': f"{rule}-{segment}-{param}",
                'original': original_threshold,
                'optimized': data['optimal_value'],
                'change': data['optimal_value'] - original_threshold
            })
    
    impact_df = pd.DataFrame(param_impact)
    impact_df = impact_df.sort_values('change', key=abs, ascending=False).head(15)
    
    plt.barh(impact_df['param'], impact_df['change'])
    plt.xlabel('Threshold Change')
    plt.ylabel('Parameter')
    plt.title('Top 15 Parameter Threshold Changes')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/statistical_threshold_changes.png')

# Run the evaluation and visualization
alerts_before, alerts_after = evaluate_overall_impact()
visualize_results(alerts_before, alerts_after)

print("\nOptimization complete! Results saved to optimized_thresholds_statistical.csv")