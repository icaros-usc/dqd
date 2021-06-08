# Computes CDFs for each archive

import re
import csv
import glob
import numpy as np
import pandas as pd
from os import path

objective_range = (0, 100)
objective_resolution = 101
archive_resolution = (200, 200)
num_cells = 1
for r in archive_resolution:
    num_cells *= r

skip_len = 200
archive_summary_filename = 'cdf.csv'
experiment_folder = '../lin_proj/logs/'
#experiment_folder = '../arm/arm_logs/'
#experiment_folder = '../lsi_clip/lsi_logs/'
archive_path = experiment_folder + '*/trial_*/archive_10000.pkl'

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

# Compile all the data
all_data = []
for archive_filename in glob.glob(archive_path):
    head, filename = path.split(archive_filename)
    head, trial_name = path.split(head)
    head, algo_name = path.split(head)
    algo_name = name_mapping[algo_name]
    _, trial_id = re.split('trial_', trial_name)
    print(algo_name, trial_id)

    df = pd.read_pickle(archive_filename)
    df_cells = sorted(df['objective'])

    n = len(df_cells)
    ptr = 0

    lo, hi = objective_range
    values = []
    for i in range(objective_resolution):
        
        thresh = (hi-lo) * (i / (objective_resolution-1)) + lo
        thresh = int(thresh+1e-9)
        
        while ptr < n and df_cells[ptr] < thresh:
            ptr += 1
        
        values.append((thresh, n-ptr))

    for thresh, cnt in values:
        cnt = (cnt / num_cells) * 100.0
        datum = [algo_name, trial_id, thresh, cnt]
        all_data.append(datum)


# Sort the data by the names in the given order.
all_data.sort(key=order_func)
all_data.insert(0,
    ['Algorithm', 'Trial', 'Objective', 'Threshold Percentage']
)

# Output the summary of summary files.
with open(archive_summary_filename, 'w') as summary_file:
    writer = csv.writer(summary_file)
    for datum in all_data:
        writer.writerow(datum)
