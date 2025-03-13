import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args

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

# Extract unique parameters for each rule and segment
param_keys = []
for rule in thresholds:
    for segment in ['IND', 'CORP']:
        rule_df = thresholds[rule]
        params = rule_df[rule_df['pop_group'] == segment]['parameter'].unique()
        for param in params:
            param_keys.append((rule, segment, param))

# Create search space for Bayesian optimization
dimensions = [Real(-0.2, 0.5, name=f"{key[0]}_{key[1]}_{key[2]}") for key in param_keys]

# Create a mapping of parameter values to scores based on thresholds
def get_score_for_value(rule, segment, param, value, threshold_adjustments):
    rule_df = thresholds[rule]
    param_df = rule_df[(rule_df['pop_group'] == segment) & (rule_df['parameter'] == param)]
    
    # Sort by threshold value
    param_df = param_df.sort_values('threshold_value')
    
    # Apply adjustments to thresholds
    key = (rule, segment, param)
    if key in threshold_adjustments:
        adjustment = threshold_adjustments[key]
        # Adjust thresholds (increase them to reduce false positives)
        param_df['threshold_value'] = param_df['threshold_value'] * (1 + adjustment)
    
    # Find the highest score where value >= threshold
    score = 0
    for _, row in param_df.iterrows():
        if value >= row['threshold_value']:
            score = row['score']
        else:
            break
    
    return score

# Objective function for Bayesian optimization
@use_named_args(dimensions)
def objective_function(**params):
    # Convert params to dictionary of adjustments
    threshold_adjustments = {}
    for key in param_keys:
        param_name = f"{key[0]}_{key[1]}_{key[2]}"
        threshold_adjustments[key] = params[param_name]
    
    # Calculate new scores for each alert
    true_positives = 0
    false_positives = 0
    true_positives_before = 0
    false_positives_before = 0
    
    for alert_id, group in alerts_data:
        is_issue = group['is_issue'].iloc[0]
        original_score = group['alert_score'].iloc[0]
        
        # Count original alerts
        if original_score >= 40:
            if is_issue == 1:
                true_positives_before += 1
            else:
                false_positives_before += 1
        
        # Calculate new score with adjusted thresholds
        new_score = 0
        for _, row in group.iterrows():
            rule = row['rule_id']
            segment = row['segment']
            param = row['param']
            value = row['value']
            
            # Get new score based on adjusted thresholds
            param_score = get_score_for_value(rule, segment, param, value, threshold_adjustments)
            new_score += param_score
        
        # Count alerts with new thresholds
        if new_score >= 40:
            if is_issue == 1:
                true_positives += 1
            else:
                false_positives += 1
    
    # Calculate fitness: maximize TP retention and minimize FP
    tp_retention = true_positives / max(1, true_positives_before)
    fp_reduction = 1 - (false_positives / max(1, false_positives_before))
    
    # Penalize heavily if we lose true positives
    if true_positives < true_positives_before:
        tp_penalty = 0.5 * (true_positives_before - true_positives) / true_positives_before
    else:
        tp_penalty = 0
    
    # Final fitness: balance between FP reduction and TP retention
    # Note: We negate the fitness because gp_minimize minimizes the objective
    fitness = -((0.7 * fp_reduction + 0.3 * tp_retention) - tp_penalty)
    
    return fitness

# Evaluate the impact of threshold adjustments
def evaluate_impact(threshold_adjustments):
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
        
        # Calculate new score with adjusted thresholds
        new_score = 0
        for _, row in group.iterrows():
            rule = row['rule_id']
            segment = row['segment']
            param = row['param']
            value = row['value']
            
            # Get new score based on adjusted thresholds
            param_score = get_score_for_value(rule, segment, param, value, threshold_adjustments)
            new_score += param_score
        
        # Count alerts with new thresholds
        if new_score >= 40:
            if is_issue == 1:
                alerts_after['TP'] += 1
            else:
                alerts_after['FP'] += 1
    
    # Print results
    print("\nImpact of Threshold Adjustments:")
    print(f"Before: {alerts_before['TP']} true positives, {alerts_before['FP']} false positives")
    print(f"After:  {alerts_after['TP']} true positives, {alerts_after['FP']} false positives")
    
    if alerts_before['FP'] > 0:
        fp_reduction = (alerts_before['FP'] - alerts_after['FP']) / alerts_before['FP'] * 100
        print(f"False Positive Reduction: {fp_reduction:.2f}%")
    
    if alerts_before['TP'] > 0:
        tp_retention = alerts_after['TP'] / alerts_before['TP'] * 100
        print(f"True Positive Retention: {tp_retention:.2f}%")
    
    return alerts_before, alerts_after

def main():
    # Run Bayesian optimization
    print("Starting Bayesian optimization...")
    result = gp_minimize(
        objective_function,
        dimensions,
        n_calls=50,  # Number of evaluations
        n_initial_points=10,  # Number of initial random points
        random_state=42,
        verbose=True
    )
    
    print(f"Optimization completed. Best fitness: {-result.fun}")
    
    # Convert best parameters to threshold adjustments
    best_adjustments = {}
    for i, key in enumerate(param_keys):
        best_adjustments[key] = result.x[i]
    
    # Save the best adjustments
    adjustments_df = pd.DataFrame([
        {'rule': key[0], 'segment': key[1], 'parameter': key[2], 'adjustment': value}
        for key, value in best_adjustments.items()
    ])
    
    # Sort by adjustment value to see which parameters were adjusted the most
    adjustments_df = adjustments_df.sort_values('adjustment', ascending=False)
    adjustments_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/optimized_adjustments_bayesian.csv', index=False)
    
    # Evaluate the impact of the best solution
    before, after = evaluate_impact(best_adjustments)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plot_convergence(result)
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/bayesian_convergence.png')
    
    # Plot top parameter importances - Fix the error here
    try:
        plt.figure(figsize=(12, 8))
        # Select only the top 5 most important dimensions based on result.space.dimension_names
        # This avoids the TypeError with Real objects
        plot_objective(result)
        plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/bayesian_parameter_importance.png')
    except Exception as e:
        print(f"Warning: Could not plot parameter importance: {e}")
        # Alternative visualization - plot the parameter adjustments directly
        plt.figure(figsize=(12, 8))
        # Get top 10 parameters by absolute adjustment value
        top_params = sorted(
            [(i, key, result.x[i]) for i, key in enumerate(param_keys)],
            key=lambda x: abs(x[2]),
            reverse=True
        )[:10]
        
        param_names = [f"{p[1][0]}_{p[1][1]}_{p[1][2]}" for p in top_params]
        param_values = [p[2] for p in top_params]
        
        plt.barh(param_names, param_values)
        plt.xlabel('Adjustment Value')
        plt.ylabel('Parameter')
        plt.title('Top 10 Parameter Adjustments')
        plt.tight_layout()
        plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/parameter_adjustments.png')
    
    # Create a bar chart showing before/after comparison
    plt.figure(figsize=(10, 6))
    labels = ['True Positives', 'False Positives']
    before_values = [before['TP'], before['FP']]
    after_values = [after['TP'], after['FP']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before_values, width, label='Before Optimization')
    rects2 = ax.bar(x + width/2, after_values, width, label='After Optimization')
    
    ax.set_ylabel('Number of Alerts')
    ax.set_title('Impact of Threshold Optimization')
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
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/bayesian_results_comparison.png')
    
    return result, best_adjustments

if __name__ == "__main__":
    result, best_adjustments = main()