import json
import csv

# Load the JSON file
file_path = '/mnt/data/experiment_results.json'
with open(file_path, 'r') as file:
    experiments = json.load(file)

# Prepare the output CSV file
output_file = '/mnt/data/experiment_means.csv'
csv_columns = ["learning_rate", "batch_size", "activation_function", "mean_accuracy_validation"]

# Process the data and calculate mean accuracies
rows = []
for experiment in experiments:
    val_accuracies = experiment.get("val_accuracies", [])
    mean_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0
    rows.append({
        "learning_rate": experiment.get("learning_rate"),
        "batch_size": experiment.get("batch_size"),
        "activation_function": experiment.get("activation_function"),
        "mean_accuracy_validation": mean_accuracy
    })

# Write to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(rows)

output_file