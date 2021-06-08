import re
import csv
import glob
from os import path

skip_len = 200
total_summary_filename = 'summary.csv'
experiment_folder = '../lin_proj/logs/'
#experiment_folder = '../arm/logs/'
#experiment_folder = '../lsi_clip/logs/'
summary_path = experiment_folder + '*/trial_*/summary.csv'

name_mapping = {
    'cma_mega': 'CMA-MEGA',
    'cma_mega_adam': 'CMA-MEGA (Adam)',
    'omg_mega': 'OMG-MEGA',
    'og_map_elites': 'OG-MAP-Elites',
    'cma_me_imp': 'CMA-ME',
    'map_elites': 'MAP-Elites',
    'map_elites_line': 'MAP-Elites (line)',
}

algo_order = [
    'CMA-MEGA (Adam)',
    'CMA-MEGA',
    'OMG-MEGA',
    'OG-MAP-Elites',
    'CMA-ME',
    'MAP-Elites (line)',
    'MAP-Elites',
]

def order_func(datum):
    return algo_order.index(datum[0])

all_data = []
for summary_file_path in glob.glob(summary_path):
    head, filename = path.split(summary_file_path)
    head, trial_name = path.split(head)
    head, algo_name = path.split(head)
    algo_name = name_mapping[algo_name]
    _, trial_id = re.split('trial_', trial_name)

    with open(summary_file_path) as summary_file:
        all_lines = list(csv.reader(summary_file))
        for cur_line in all_lines[skip_len::skip_len]:
            datum = [algo_name, trial_id] + cur_line
            print(datum)
            all_data.append(datum)

# Sort the data by the names in the given order.
all_data.sort(key=order_func)
all_data.insert(0, 
    ['Algorithm', 'Trial', 'Iteration', 'QD-Score', 'Coverage', 'Best Solution', 'Average']
)

# Output the summary of summary files.
with open(total_summary_filename, 'w') as summary_file:
    writer = csv.writer(summary_file)    
    for datum in all_data:
        writer.writerow(datum)
