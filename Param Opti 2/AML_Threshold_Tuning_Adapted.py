import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Load the generated data
data_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df = pd.read_csv(data_path)

# Load the thresholds
excel_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
thresholds = {}
for rule in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    thresholds[rule] = pd.read_excel(excel_path, sheet_name=rule)

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Create alert_id to group by alert
df['alert_id'] = df['alert_date'] + '_' + df['account_number'].astype(str)

# Analyze the distribution of alerts
alert_counts = df.groupby('alert_id')['is_issue'].max().reset_index()
total_alerts = len(alert_counts)
true_positives = alert_counts['is_issue'].sum()
false_positives = total_alerts - true_positives

print(f"\nTotal alerts: {total_alerts}")
print(f"True positives: {true_positives} ({true_positives/total_alerts:.2%})")
print(f"False positives: {false_positives} ({false_positives/total_alerts:.2%})")

# Analyze the distribution of parameters
print("\nParameter distribution:")
param_counts = df['param'].value_counts()
print(param_counts.head(10))

# Function to evaluate threshold performance
def evaluate_threshold(rule_id, segment, parameter, threshold_value):
    """
    Evaluate the performance of a specific threshold value for a parameter
    """
    # Filter data for this rule-segment-parameter
    param_data = df[(df['rule_id'] == rule_id) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parameter)]
    
    if param_data.empty:
        return {
            'threshold': threshold_value,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0
        }
    
    # Group by alert_id to avoid double counting
    alerts = param_data.groupby('alert_id')['is_issue'].max().reset_index()
    alerts['predicted'] = 0
    
    # Apply threshold
    for alert_id in alerts['alert_id']:
        alert_data = param_data[param_data['alert_id'] == alert_id]
        if any(alert_data['value'] >= threshold_value):
            alerts.loc[alerts['alert_id'] == alert_id, 'predicted'] = 1
    
    # Calculate metrics
    tp = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 1))
    fp = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 1))
    tn = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 0))
    fn = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'threshold': threshold_value,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity
    }

# Function to find optimal threshold for a parameter
def find_optimal_threshold(rule_id, segment, parameter, metric='f1'):
    """
    Find the optimal threshold value for a parameter based on a specified metric
    """
    # Filter data for this rule-segment-parameter
    param_data = df[(df['rule_id'] == rule_id) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parameter)]
    
    if param_data.empty:
        print(f"No data for {rule_id}-{segment}-{parameter}")
        return None, None
    
    # Get unique values and sort them
    values = sorted(param_data['value'].unique())
    
    # If too many values, sample a reasonable number
    if len(values) > 50:
        values = np.percentile(param_data['value'], np.linspace(0, 100, 50))
    
    # Ensure all values are integers
    values = [int(round(v)) for v in values]
    values = sorted(list(set(values)))  # Remove duplicates
    
    # Evaluate each threshold
    results = []
    for threshold in values:
        metrics = evaluate_threshold(rule_id, segment, parameter, threshold)
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold based on specified metric
    if not results_df.empty:
        optimal_idx = results_df[metric].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        optimal_metrics = results_df.loc[optimal_idx]
        
        return optimal_threshold, optimal_metrics
    
    return None, None

# Function to plot threshold performance curves
def plot_threshold_curves(rule_id, segment, parameter):
    """
    Plot performance curves for different threshold values
    """
    # Filter data for this rule-segment-parameter
    param_data = df[(df['rule_id'] == rule_id) & 
                    (df['segment'] == segment) & 
                    (df['param'] == parameter)]
    
    if param_data.empty:
        print(f"No data for {rule_id}-{segment}-{parameter}")
        return
    
    # Get unique values and sort them
    values = sorted(param_data['value'].unique())
    
    # If too many values, sample a reasonable number
    if len(values) > 50:
        values = np.percentile(param_data['value'], np.linspace(0, 100, 50))
    
    # Ensure all values are integers
    values = [int(round(v)) for v in values]
    values = sorted(list(set(values)))  # Remove duplicates
    
    # Evaluate each threshold
    results = []
    for threshold in values:
        metrics = evaluate_threshold(rule_id, segment, parameter, threshold)
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision')
    ax.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall')
    ax.plot(results_df['threshold'], results_df['f1'], 'r-', label='F1 Score')
    ax.plot(results_df['threshold'], results_df['specificity'], 'c-', label='Specificity')
    
    # Find optimal threshold based on F1 score
    optimal_idx = results_df['f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    # Add vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, color='k', linestyle='--', 
               label=f'Optimal Threshold: {optimal_threshold}')
    
    ax.set_xlabel('Threshold Value')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Performance Curves for {rule_id}-{segment}-{parameter}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/Users/paulconerardy/Documents/AML/Param Opti 2/threshold_curves_{rule_id}_{segment}_{parameter}.png')
    plt.close()
    
    return results_df

# Extract unique rule-segment-parameter combinations
param_combinations = []
for rule in thresholds:
    for segment in ['IND', 'CORP']:
        rule_df = thresholds[rule]
        params = rule_df[rule_df['pop_group'] == segment]['parameter'].unique()
        for param in params:
            param_combinations.append((rule, segment, param))

print(f"\nTotal parameter combinations to analyze: {len(param_combinations)}")

# Find optimal thresholds for all parameters
optimal_thresholds = []

for rule, segment, param in param_combinations:
    print(f"Analyzing {rule}-{segment}-{param}...")
    
    # Find optimal threshold
    threshold, metrics = find_optimal_threshold(rule, segment, param)
    
    if threshold is not None:
        # Plot threshold curves
        results_df = plot_threshold_curves(rule, segment, param)
        
        # Store results
        optimal_thresholds.append({
            'rule': rule,
            'segment': segment,
            'parameter': param,
            'optimal_threshold': threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'specificity': metrics['specificity'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn']
        })

# Convert to DataFrame and save
optimal_df = pd.DataFrame(optimal_thresholds)
optimal_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/optimal_thresholds.csv', index=False)

# Display top parameters by F1 score
print("\nTop parameters by F1 score:")
print(optimal_df.sort_values('f1', ascending=False).head(10))

# Plot the distribution of optimal thresholds
plt.figure(figsize=(12, 8))
sns.histplot(optimal_df['optimal_threshold'], bins=20, kde=True)
plt.title('Distribution of Optimal Thresholds')
plt.xlabel('Threshold Value')
plt.ylabel('Frequency')
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/optimal_threshold_distribution.png')
plt.close()

# Plot F1 scores for each parameter
plt.figure(figsize=(15, 10))
top_params = optimal_df.sort_values('f1', ascending=False).head(20)
sns.barplot(x='f1', y='parameter', hue='rule', data=top_params)
plt.title('F1 Scores for Top 20 Parameters')
plt.xlabel('F1 Score')
plt.ylabel('Parameter')
plt.tight_layout()
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/top_parameters_f1_scores.png')
plt.close()

# Function to evaluate the overall impact of optimized thresholds
def evaluate_optimized_system():
    """
    Evaluate the performance of the system with optimized thresholds
    """
    # Create a dictionary of optimal thresholds
    threshold_dict = {}
    for _, row in optimal_df.iterrows():
        key = (row['rule'], row['segment'], row['parameter'])
        threshold_dict[key] = row['optimal_threshold']
    
    # Group data by alert
    alerts = df.groupby('alert_id')['is_issue'].max().reset_index()
    alerts['predicted'] = 0
    
    # Apply optimized thresholds
    for alert_id in alerts['alert_id']:
        alert_data = df[df['alert_id'] == alert_id]
        
        # Check if any parameter exceeds its threshold
        for _, row in alert_data.iterrows():
            key = (row['rule_id'], row['segment'], row['param'])
            if key in threshold_dict:
                if row['value'] >= threshold_dict[key]:
                    alerts.loc[alerts['alert_id'] == alert_id, 'predicted'] = 1
                    break
    
    # Calculate metrics
    tp = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 1))
    fp = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 1))
    tn = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 0))
    fn = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\nOverall System Performance with Optimized Thresholds:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(alerts['is_issue'], alerts['predicted'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Alerted', 'Alerted'],
                yticklabels=['Not an Issue', 'Issue'])
    plt.title('Confusion Matrix for Optimized System')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/optimized_system_confusion_matrix.png')
    plt.close()
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity
    }

# Evaluate the optimized system
optimized_metrics = evaluate_optimized_system()

# Compare with baseline (current thresholds)
def evaluate_baseline_system():
    """
    Evaluate the performance of the system with current thresholds
    """
    # Group data by alert
    alerts = df.groupby('alert_id')['is_issue'].max().reset_index()
    alerts['predicted'] = 0
    
    # For baseline, we use the existing alert_score
    alert_scores = df.groupby('alert_id')['alert_score'].max().reset_index()
    
    # Merge alert scores with alerts
    alerts = alerts.merge(alert_scores, on='alert_id', how='left')
    
    # Apply threshold of 40 (standard alert threshold)
    alerts['predicted'] = (alerts['alert_score'] >= 40).astype(int)
    
    # Calculate metrics
    tp = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 1))
    fp = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 1))
    tn = sum((alerts['is_issue'] == 0) & (alerts['predicted'] == 0))
    fn = sum((alerts['is_issue'] == 1) & (alerts['predicted'] == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\nBaseline System Performance:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(alerts['is_issue'], alerts['predicted'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Alerted', 'Alerted'],
                yticklabels=['Not an Issue', 'Issue'])
    plt.title('Confusion Matrix for Baseline System')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/baseline_system_confusion_matrix.png')
    plt.close()
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity
    }

# Evaluate the baseline system
baseline_metrics = evaluate_baseline_system()

# Compare baseline and optimized metrics
comparison = pd.DataFrame({
    'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives',
               'Precision', 'Recall', 'F1 Score', 'Specificity'],
    'Baseline': [baseline_metrics['tp'], baseline_metrics['fp'], 
                 baseline_metrics['tn'], baseline_metrics['fn'],
                 baseline_metrics['precision'], baseline_metrics['recall'], 
                 baseline_metrics['f1'], baseline_metrics['specificity']],
    'Optimized': [optimized_metrics['tp'], optimized_metrics['fp'], 
                  optimized_metrics['tn'], optimized_metrics['fn'],
                  optimized_metrics['precision'], optimized_metrics['recall'], 
                  optimized_metrics['f1'], optimized_metrics['specificity']]
})

# Calculate improvement
comparison['Improvement'] = comparison['Optimized'] - comparison['Baseline']
comparison['Percent Change'] = (comparison['Improvement'] / comparison['Baseline']) * 100

# Format percentage columns
for col in ['Precision', 'Recall', 'F1 Score', 'Specificity']:
    idx = comparison['Metric'] == col
    comparison.loc[idx, 'Baseline'] = comparison.loc[idx, 'Baseline'].apply(lambda x: f"{x:.4f}")
    comparison.loc[idx, 'Optimized'] = comparison.loc[idx, 'Optimized'].apply(lambda x: f"{x:.4f}")
    comparison.loc[idx, 'Improvement'] = comparison.loc[idx, 'Improvement'].apply(lambda x: f"{x:.4f}")
    comparison.loc[idx, 'Percent Change'] = comparison.loc[idx, 'Percent Change'].apply(lambda x: f"{x:.2f}%")

print("\nComparison of Baseline and Optimized Systems:")
print(comparison)

# Save comparison to CSV
comparison.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/system_comparison.csv', index=False)

# Create a bar chart comparing baseline and optimized metrics
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Specificity']
baseline_values = [float(comparison.loc[comparison['Metric'] == m, 'Baseline'].values[0]) for m in metrics_to_plot]
optimized_values = [float(comparison.loc[comparison['Metric'] == m, 'Optimized'].values[0]) for m in metrics_to_plot]

plt.figure(figsize=(12, 8))
x = np.arange(len(metrics_to_plot))
width = 0.35

plt.bar(x - width/2, baseline_values, width, label='Baseline')
plt.bar(x + width/2, optimized_values, width, label='Optimized')

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Comparison of Baseline and Optimized Systems')
plt.xticks(x, metrics_to_plot)
plt.legend()
plt.grid(True, axis='y')

# Add value labels
for i, v in enumerate(baseline_values):
    plt.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center')
    
for i, v in enumerate(optimized_values):
    plt.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center')

plt.tight_layout()
plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/system_comparison_chart.png')
plt.close()

# Create a function to update the thresholds Excel file with optimized values
def update_thresholds_excel():
    """
    Update the thresholds Excel file with optimized values
    """
    # Create a copy of the original thresholds
    updated_thresholds = {}
    for rule in thresholds:
        updated_thresholds[rule] = thresholds[rule].copy()
    
    # Update with optimized thresholds
    for _, row in optimal_df.iterrows():
        rule = row['rule']
        segment = row['segment']
        parameter = row['parameter']
        optimal_threshold = row['optimal_threshold']
        
        # Find the rows to update
        mask = (updated_thresholds[rule]['pop_group'] == segment) & (updated_thresholds[rule]['parameter'] == parameter)
        
        # Update threshold values
        if any(mask):
            # Get the minimum threshold value for this parameter
            min_threshold = updated_thresholds[rule].loc[mask, 'threshold_value'].min()
            
            # Calculate the adjustment factor
            adjustment = optimal_threshold - min_threshold
            
            # Apply the adjustment to all threshold values for this parameter
            updated_thresholds[rule].loc[mask, 'threshold_value'] += adjustment
    
    # Save updated thresholds to a new Excel file
    with pd.ExcelWriter('/Users/paulconerardy/Documents/AML/Param Opti 2/optimized_aml_thresholds.xlsx') as writer:
        for rule in updated_thresholds:
            updated_thresholds[rule].to_excel(writer, sheet_name=rule, index=False)
    
    print("\nUpdated thresholds saved to optimized_aml_thresholds.xlsx")

# Update the thresholds Excel file
update_thresholds_excel()

print("\nAnalysis complete!")