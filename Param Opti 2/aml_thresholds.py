import pandas as pd
import numpy as np
import os

# Define the rules and population groups
rules = ['AML-TUA', 'AML-MNT', 'AML-CHG', 'AML-AUG', 'AML-FTF']
pop_groups = ['IND', 'CORP']

# Create a directory to save the Excel file
output_path = '/Users/paulconerardy/Documents/AML/Param Opti 2'
os.makedirs(output_path, exist_ok=True)

# Create an Excel writer object
excel_path = os.path.join(output_path, 'aml_thresholds.xlsx')
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# For each rule, create a tab with thresholds and scores
for rule in rules:
    # Define parameters based on the rule
    if rule == 'AML-TUA':
        parameters = ['activity_value', 'transaction_count', 'avg_transaction_size']
    elif rule == 'AML-MNT':
        parameters = ['activity_value', 'transaction_frequency', 'destination_count']
    elif rule == 'AML-CHG':
        parameters = ['activity_ratio_change', 'volume_increase', 'pattern_deviation']
    elif rule == 'AML-AUG':
        parameters = ['consecutive_inc', 'peak_deviation', 'seasonal_factor']
    elif rule == 'AML-FTF':
        parameters = ['foreign_transaction_ratio', 'high_risk_country_count', 'transaction_velocity']
    
    # Create dataframe for this rule
    data = []
    
    for pop_group in pop_groups:
        for param in parameters:
            # Generate different threshold values based on parameter and population group
            if 'value' in param or 'size' in param:
                if pop_group == 'IND':
                    thresholds = [5000, 10000, 20000, 30000, 50000]
                else:  # CORP
                    thresholds = [20000, 50000, 100000, 200000, 500000]
            elif 'count' in param or 'inc' in param:
                if pop_group == 'IND':
                    thresholds = [2, 3, 5, 7, 10]
                else:  # CORP
                    thresholds = [5, 10, 15, 20, 30]
            elif 'ratio' in param or 'factor' in param:
                if pop_group == 'IND':
                    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
                else:  # CORP
                    thresholds = [0.3, 0.7, 1.2, 1.8, 2.5]
            else:
                if pop_group == 'IND':
                    thresholds = [1, 2, 3, 4, 5]
                else:  # CORP
                    thresholds = [2, 4, 6, 8, 10]
            
            # Generate linearly correlated scores (higher value = higher score)
            scores = [5, 10, 15, 20, 25]
            
            # Add rows to the data
            for threshold, score in zip(thresholds, scores):
                data.append([pop_group, param, threshold, score])
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data, columns=['pop_group', 'parameter', 'threshold_value', 'score'])
    df.to_excel(writer, sheet_name=rule, index=False)
    
    # Adjust column widths
    worksheet = writer.sheets[rule]
    worksheet.set_column('A:A', 10)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 10)

# Save the Excel file
writer.close()

print(f"Excel file created at: {excel_path}")