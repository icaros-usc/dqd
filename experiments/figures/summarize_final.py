# A script for generating final archive summary stats.
# Used for statistical tests and generating tables.

import re
import csv
import glob
from os import path

# Change for each experiment.
exp_name = 'lin_proj'

total_summary_filename = "summary.csv"
experiment_folder = f"../{exp_name}/logs/"
summary_path = experiment_folder + "*/trial_*/summary.csv"

all_data = [['Algorithm', 'Trial', 'QD-Score', 'Coverage', 'Maximum', 'Average']]
for summary_file_path in glob.glob(summary_path):
    head, filename = path.split(summary_file_path)
    head, trial_name = path.split(head)
    head, algo_name = path.split(head)
    _, trial_id = re.split('trial_', trial_name)

    with open(summary_file_path) as summary_file:
        all_lines = list(csv.reader(summary_file))
        datum = [algo_name, trial_id] + all_lines[-1][1:]
        print(datum)
        all_data.append(datum)

# Output the summary of summary files.
with open(total_summary_filename, 'w') as summary_file:
    writer = csv.writer(summary_file)    
    for datum in all_data:
        writer.writerow(datum)
