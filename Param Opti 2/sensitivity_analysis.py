import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
alerts = df.groupby('alert_id')['is_issue'].max().reset_index()
alerts_data = df.groupby('alert_id')

# Extract unique parameters for each rule and segment
param_keys = []
for rule in thresholds:
    for segment in ['IND', 'CORP']:
        rule_df = thresholds[rule]
        params = rule_df[rule_df['pop_group'] == segment]['parameter'].unique()
        for param in params:
            param_keys.append((rule, segment, param))

# Function to calculate alert scores with adjusted thresholds
def calculate_alert_scores(threshold_adjustments):
    results = {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0
    }
    
    # Process each alert
    for alert_id, group in alerts_data:
        is_issue = group['is_issue'].iloc[0]
        original_score = group['alert_score'].iloc[0] if 'alert_score' in group.columns else sum(group['score'])
        
        # Calculate new score with adjusted thresholds
        new_score = 0
        for _, row in group.iterrows():
            rule = row['rule_id']
            segment = row['segment']
            param = row['param']
            value = row['value']
            
            # Get the original threshold and score mapping
            rule_df = thresholds[rule]
            threshold_df = rule_df[(rule_df['pop_group'] == segment) & (rule_df['parameter'] == param)]
            
            # Apply threshold adjustment if specified
            key = (rule, segment, param)
            if key in threshold_adjustments:
                adjustment = threshold_adjustments[key]
                # Adjust the threshold by adding the integer adjustment
                adjusted_thresholds = threshold_df.copy()
                adjusted_thresholds['threshold_value'] = adjusted_thresholds['threshold_value'] + adjustment
            else:
                adjusted_thresholds = threshold_df
            
            # Find the appropriate score based on the value and adjusted thresholds
            param_score = 0
            for _, t_row in adjusted_thresholds.sort_values('threshold_value').iterrows():
                if value >= t_row['threshold_value']:
                    param_score = t_row['score']
                else:
                    break
            
            new_score += param_score
        
        # Determine if this is a true/false positive/negative
        if new_score >= 40:  # Alert threshold
            if is_issue == 1:
                results['TP'] += 1
            else:
                results['FP'] += 1
        else:
            if is_issue == 1:
                results['FN'] += 1
            else:
                results['TN'] += 1
    
    # Calculate metrics
    if results['TP'] + results['FP'] > 0:
        precision = results['TP'] / (results['TP'] + results['FP'])
    else:
        precision = 0
        
    if results['TP'] + results['FN'] > 0:
        recall = results['TP'] / (results['TP'] + results['FN'])
    else:
        recall = 0
        
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return {
        'TP': results['TP'],
        'FP': results['FP'],
        'TN': results['TN'],
        'FN': results['FN'],
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Calculate baseline performance (no adjustments)
baseline = calculate_alert_scores({})
print("Baseline Performance:")
print(f"True Positives: {baseline['TP']}")
print(f"False Positives: {baseline['FP']}")
print(f"True Negatives: {baseline['TN']}")
print(f"False Negatives: {baseline['FN']}")
print(f"Precision: {baseline['precision']:.4f}")
print(f"Recall: {baseline['recall']:.4f}")
print(f"F1 Score: {baseline['f1']:.4f}")

# Perform sensitivity analysis for each parameter
sensitivity_results = []

# Range of adjustments to test for each parameter
adjustment_range = range(-10, 11)  # -10 to +10

print("\nPerforming sensitivity analysis...")
for rule, segment, param in tqdm(param_keys):
    param_sensitivity = []
    
    for adjustment in adjustment_range:
        # Apply adjustment to just this parameter
        threshold_adjustments = {(rule, segment, param): adjustment}
        
        # Calculate performance metrics
        metrics = calculate_alert_scores(threshold_adjustments)
        
        param_sensitivity.append({
            'rule': rule,
            'segment': segment,
            'parameter': param,
            'adjustment': adjustment,
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    sensitivity_results.extend(param_sensitivity)

# Convert results to DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results)

# Save the full sensitivity analysis results
sensitivity_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/sensitivity_analysis_results.csv', index=False)

# Calculate the impact of each parameter on F1 score
param_impact = []
for (rule, segment, param) in param_keys:
    param_data = sensitivity_df[(sensitivity_df['rule'] == rule) & 
                               (sensitivity_df['segment'] == segment) & 
                               (sensitivity_df['parameter'] == param)]
    
    # Calculate the range of F1 scores for this parameter
    f1_range = param_data['f1'].max() - param_data['f1'].min()
    
    # Find the optimal adjustment (highest F1 score)
    optimal_idx = param_data['f1'].idxmax()
    optimal_adjustment = param_data.loc[optimal_idx, 'adjustment']
    optimal_f1 = param_data.loc[optimal_idx, 'f1']
    
    param_impact.append({
        'rule': rule,
        'segment': segment,
        'parameter': param,
        'f1_range': f1_range,
        'optimal_adjustment': optimal_adjustment,
        'optimal_f1': optimal_f1
    })

# Convert to DataFrame and sort by impact
impact_df = pd.DataFrame(param_impact)
impact_df = impact_df.sort_values('f1_range', ascending=False)

# Save the parameter impact analysis
impact_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/parameter_impact_analysis.csv', index=False)

# Print the top 10 most impactful parameters
print("\nTop 10 Most Impactful Parameters:")
print(impact_df.head(10))

# Visualize the sensitivity of the top 5 parameters
plt.figure(figsize=(15, 10))

top_params = impact_df.head(5)
for i, row in top_params.iterrows():
    rule = row['rule']
    segment = row['segment']
    param = row['parameter']
    
    # Get sensitivity data for this parameter
    param_data = sensitivity_df[(sensitivity_df['rule'] == rule) & 
                               (sensitivity_df['segment'] == segment) & 
                               (sensitivity_df['parameter'] == param)]
    
    # Plot F1 score vs adjustment
    plt.plot(param_data['adjustment'], param_data['f1'], 
             marker='o', label=f"{rule}-{segment}-{param}")

plt.axhline(y=baseline['f1'], color='r', linestyle='--', label='Baseline F1')
plt.xlabel('Threshold Adjustment')
plt.ylabel('F1 Score')
plt.title('Sensitivity Analysis: Impact of Threshold Adjustments on F1 Score')
plt.legend()
plt.grid(True)
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/sensitivity_top5_parameters.png')

# Create a heatmap of parameter adjustments vs metrics for the most impactful parameter
top_param = impact_df.iloc[0]
top_rule = top_param['rule']
top_segment = top_param['segment']
top_parameter = top_param['parameter']

top_param_data = sensitivity_df[(sensitivity_df['rule'] == top_rule) & 
                               (sensitivity_df['segment'] == top_segment) & 
                               (sensitivity_df['parameter'] == top_parameter)]

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Detailed Sensitivity Analysis for {top_rule}-{top_segment}-{top_parameter}', fontsize=16)

# Plot TP and FP
axes[0, 0].plot(top_param_data['adjustment'], top_param_data['TP'], 'g-o', label='True Positives')
axes[0, 0].plot(top_param_data['adjustment'], top_param_data['FP'], 'r-o', label='False Positives')
axes[0, 0].set_xlabel('Threshold Adjustment')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('TP and FP vs Threshold Adjustment')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot Precision
axes[0, 1].plot(top_param_data['adjustment'], top_param_data['precision'], 'b-o')
axes[0, 1].set_xlabel('Threshold Adjustment')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision vs Threshold Adjustment')
axes[0, 1].grid(True)

# Plot Recall
axes[1, 0].plot(top_param_data['adjustment'], top_param_data['recall'], 'm-o')
axes[1, 0].set_xlabel('Threshold Adjustment')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_title('Recall vs Threshold Adjustment')
axes[1, 0].grid(True)

# Plot F1 Score
axes[1, 1].plot(top_param_data['adjustment'], top_param_data['f1'], 'c-o')
axes[1, 1].set_xlabel('Threshold Adjustment')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score vs Threshold Adjustment')
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/top_parameter_detailed_analysis.png')

# Create a combined optimal threshold configuration
print("\nCreating optimal threshold configuration based on sensitivity analysis...")
optimal_adjustments = {}
for _, row in impact_df.iterrows():
    key = (row['rule'], row['segment'], row['parameter'])
    optimal_adjustments[key] = int(row['optimal_adjustment'])

# Evaluate the combined optimal configuration
optimal_metrics = calculate_alert_scores(optimal_adjustments)

print("\nOptimal Configuration Performance:")
print(f"True Positives: {optimal_metrics['TP']} (Baseline: {baseline['TP']})")
print(f"False Positives: {optimal_metrics['FP']} (Baseline: {baseline['FP']})")
print(f"True Negatives: {optimal_metrics['TN']} (Baseline: {baseline['TN']})")
print(f"False Negatives: {optimal_metrics['FN']} (Baseline: {baseline['FN']})")
print(f"Precision: {optimal_metrics['precision']:.4f} (Baseline: {baseline['precision']:.4f})")
print(f"Recall: {optimal_metrics['recall']:.4f} (Baseline: {baseline['recall']:.4f})")
print(f"F1 Score: {optimal_metrics['f1']:.4f} (Baseline: {baseline['f1']:.4f})")

# Save the optimal adjustments
optimal_df = pd.DataFrame([
    {'rule': key[0], 'segment': key[1], 'parameter': key[2], 'adjustment': value}
    for key, value in optimal_adjustments.items()
])
optimal_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/optimal_threshold_adjustments.csv', index=False)

print("\nAnalysis complete! Results saved to files.")