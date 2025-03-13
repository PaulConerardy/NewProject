import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Load the thresholds from the Excel file
excel_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/aml_thresholds.xlsx'
thresholds = {}

# Read each sheet (rule) from the Excel file
for rule in ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']:
    thresholds[rule] = pd.read_excel(excel_path, sheet_name=rule)

# Function to get parameter value that would result in a specific score
def get_param_value_for_score(rule, pop_group, parameter, target_score):
    rule_df = thresholds[rule]
    param_df = rule_df[(rule_df['pop_group'] == pop_group) & (rule_df['parameter'] == parameter)]
    
    # Find the threshold that corresponds to the target score
    threshold_row = param_df[param_df['score'] == target_score]
    
    if not threshold_row.empty:
        return threshold_row.iloc[0]['threshold_value']
    else:
        # If exact score not found, get the closest one
        closest_score = min(param_df['score'], key=lambda x: abs(x - target_score))
        threshold_row = param_df[param_df['score'] == closest_score]
        return threshold_row.iloc[0]['threshold_value']

# Function to generate a random date within the last year
def random_date():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date.strftime('%Y-%m-%d')

# Function to generate synthetic AML data
def generate_aml_data(num_alerts=2000, min_score_threshold=40, true_positive_rate=0.05):
    data = []
    
    # Define account numbers and segments
    account_numbers = list(range(1000, 9999))
    segments = ['IND', 'CORP']
    
    # Generate alerts
    for _ in range(num_alerts):
        # Each alert has a random account number and segment
        account_number = random.choice(account_numbers)
        segment = random.choice(segments)
        alert_date = random_date()
        
        # Decide if this will be a true positive (is_issue=1) or false positive (is_issue=0)
        # based on the specified true positive rate
        is_issue = 1 if random.random() < true_positive_rate else 0
        
        # Randomly select 1-3 rules that will contribute to this alert
        num_rules = random.randint(1, 3)
        selected_rules = random.sample(['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF'], num_rules)
        
        alert_score = 0
        alert_rows = []
        
        # For each rule, generate parameter values and scores
        for rule in selected_rules:
            rule_df = thresholds[rule]
            
            # Get parameters for this rule
            parameters = rule_df[rule_df['pop_group'] == segment]['parameter'].unique()
            
            # Randomly select one parameter for this rule
            param = random.choice(parameters)
            
            # Determine score for this parameter (higher for true positives)
            if is_issue:
                score = random.choice([15, 20, 25])  # Higher scores for true positives
            else:
                score = random.choice([5, 10, 15])   # Lower scores for false positives
            
            # Get a parameter value that would result in this score
            value = get_param_value_for_score(rule, segment, param, score)
            
            # Add some random variation to the value
            if isinstance(value, (int, float)):
                if 'ratio' in param or 'factor' in param:
                    value = value * random.uniform(0.9, 1.1)  # 10% variation for ratios
                else:
                    value = value * random.uniform(0.8, 1.2)  # 20% variation for other values
            
            # Add to the alert score
            alert_score += score
            
            # Create a row for this rule detection
            alert_rows.append({
                'alert_date': alert_date,
                'account_number': account_number,
                'segment': segment,
                'rule_id': rule,
                'param': param,
                'value': value,
                'score': score,
                'alert_score': None,  # Will fill in later
                'is_issue': is_issue
            })
        
        # Only keep alerts that meet the minimum score threshold
        if alert_score >= min_score_threshold:
            # Update all rows with the total alert score
            for row in alert_rows:
                row['alert_score'] = alert_score
                data.append(row)
    
    return pd.DataFrame(data)

# Generate the data with 5% true positive rate
df = generate_aml_data(num_alerts=10000, min_score_threshold=40, true_positive_rate=0.01)

# Sort by alert_date, account_number
df = df.sort_values(['alert_date', 'account_number'])

# Save to CSV
output_path = '/Users/paulconerardy/Documents/AML/Param Opti 2/generated_data.csv'
df.to_csv(output_path, index=False)

print(f"Generated AML data saved to: {output_path}")
print(f"Total alerts: {len(df['alert_date'].unique())}")
print(f"Total rule detections: {len(df)}")
print(f"True positives: {df[df['is_issue'] == 1]['alert_date'].nunique()}")
print(f"False positives: {df[df['is_issue'] == 0]['alert_date'].nunique()}")
print(f"True positive rate: {df[df['is_issue'] == 1]['alert_date'].nunique() / len(df['alert_date'].unique()):.2%}")