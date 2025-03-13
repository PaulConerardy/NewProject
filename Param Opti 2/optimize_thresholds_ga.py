import pandas as pd
import numpy as np
import random
import os
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

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
        # Adjust thresholds with integer adjustments
        param_df['threshold_value'] = param_df['threshold_value'] + adjustment
    
    # Find the highest score where value >= threshold
    score = 0
    for _, row in param_df.iterrows():
        if value >= row['threshold_value']:
            score = row['score']
        else:
            break
    
    return score

# Evaluate a set of threshold adjustments
def evaluate_thresholds(individual):
    # Convert individual to dictionary of adjustments
    threshold_adjustments = {}
    for i, key in enumerate(param_keys):
        threshold_adjustments[key] = individual[i]
    
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
    fitness = (0.7 * fp_reduction + 0.3 * tp_retention) - tp_penalty
    
    return (fitness,)

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define genes as integers between -5 and 10 instead of floats
toolbox.register("attr_int", random.randint, -5, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(param_keys))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("evaluate", evaluate_thresholds)
toolbox.register("mate", tools.cxTwoPoint)  # Changed to two-point crossover for integers
toolbox.register("mutate", tools.mutUniformInt, low=-5, up=10, indpb=0.2)  # Integer mutation
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the algorithm
def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Run for 30 generations
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, 
                                  stats=stats, halloffame=hof, verbose=True)
    
    # Get the best solution
    best = hof[0]
    
    # Convert best individual to threshold adjustments
    best_adjustments = {}
    for i, key in enumerate(param_keys):
        best_adjustments[key] = best[i]
    
    # Save the best adjustments
    adjustments_df = pd.DataFrame([
        {'rule': key[0], 'segment': key[1], 'parameter': key[2], 'adjustment': value}
        for key, value in best_adjustments.items()
    ])
    
    # Sort by adjustment value to see which parameters were adjusted the most
    adjustments_df = adjustments_df.sort_values('adjustment', ascending=False)
    adjustments_df.to_csv('/Users/paulconerardy/Documents/AML/Param Opti 2/optimized_adjustments.csv', index=False)
    
    # Evaluate the impact of the best solution
    evaluate_impact(best_adjustments)
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    gen = range(len(log.select('max')))
    plt.plot(gen, log.select('avg'), 'k-', label='Average Fitness')
    plt.plot(gen, log.select('max'), 'r-', label='Best Fitness')
    plt.title('Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('/Users/paulconerardy/Documents/AML/Param Opti 2/fitness_evolution.png')
    
    return pop, log, hof

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

if __name__ == "__main__":
    pop, log, hof = main()