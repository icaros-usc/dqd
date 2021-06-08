# A small script for generating a collage of faces from the QD-Archive

import matplotlib
import matplotlib.font_manager
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = 'serif'
matplotlib.rcParams["font.serif"] = 'Palatino'
matplotlib.rc('font', size=20)

import torch
import numpy as np
import pandas as pd
from stylegan_models import g_all, g_synthesis, g_mapping

from PIL import Image

# Note that only final archives encode latent codes.
archive_filename = 'logs/cma_mega_adam/trial_0/archive_10000.pkl'

# min and max index for rows then columns (row major).
# The archive is shape (200, 200) indexed from [0, 200).
archive_dims = (200, 200)
archive_index_range = ((90, 200), (90, 200))
# Measure ranges
measure_ranges = ((0,6), (0,6))
# Controls that x rows and y columns are generated
# Images are "evenly" (as possible) sampled based on this criteria
picture_frequency = (8, 5) 

# Use the CPU while we are running exps.
device = "cpu"

g_synthesis.eval()
g_synthesis.to(device)
for p in g_synthesis.parameters():
    p.requires_grad_(False)

# Read the archive from the log (pickle file)
df = pd.read_pickle(archive_filename)

imgs = []
for j in reversed(range(picture_frequency[1])):
    for i in range(picture_frequency[0]):
        
        delta_i = archive_index_range[0][1] - archive_index_range[0][0]
        delta_j = archive_index_range[1][1] - archive_index_range[1][0]
        index_i_lower = int(delta_i * i / picture_frequency[0] + archive_index_range[0][0])
        index_i_upper = int(delta_i * (i+1) / picture_frequency[0] + archive_index_range[0][0])
        index_j_lower = int(delta_j * j / picture_frequency[1] + archive_index_range[1][0])
        index_j_upper = int(delta_j * (j+1) / picture_frequency[1] + archive_index_range[1][0])
        print(i, j, index_i_lower, index_i_upper, index_j_lower, index_j_upper)

        query_string = f"{index_i_lower} <= index_0 & index_0 <= {index_i_upper} &"
        query_string += f"{index_j_lower} <= index_1 & index_1 <= {index_j_upper}" 
        print(query_string)
        df_cell = df.query(query_string)
        sol = df_cell.iloc[df_cell['objective'].argmax()]
        print(sol)

        latent_code = torch.tensor(sol[5:].values, dtype=torch.float32, device=device)
        latents = torch.nn.Parameter(latent_code, requires_grad=False)
        dlatents = latents.repeat(1,18,1)

        img = g_synthesis(dlatents)
        img = (img.clamp(-1, 1) + 1) / 2.0 # Normalize from [0,1]
        img = img[0].detach().cpu()
        imgs.append(img)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

plt.figure(figsize=(16,10))
img_grid = make_grid(imgs, nrow=picture_frequency[0], padding=0)
img_grid = np.transpose(img_grid.cpu().numpy(), (1,2,0))
plt.imshow(img_grid)

plt.xlabel("A man with blue eyes.")
plt.ylabel("A person with red hair.")

def create_archive_tick_labels(axis_range, measure_range, dim, num_ticks):
    low_pos = axis_range[0] / dim
    high_pos = axis_range[1] / dim

    tick_offset = [
        (high_pos - low_pos) * (p / num_ticks) + low_pos
        for p in range(num_ticks+1) 
    ]
    ticklabels = [
        round((measure_range[1]-measure_range[0]) * p + measure_range[0], 2)
        for p in tick_offset
    ]
    return ticklabels

num_x_ticks = 6
num_y_ticks = 6
x_ticklabels = create_archive_tick_labels(archive_index_range[0], 
                    measure_ranges[0], archive_dims[0], num_x_ticks)
y_ticklabels = create_archive_tick_labels(archive_index_range[1],
                    measure_ranges[1], archive_dims[1], num_y_ticks)
y_ticklabels.reverse()

x_tick_range = img_grid.shape[1]
x_ticks = np.arange(0, x_tick_range+1e-9, step=x_tick_range/num_x_ticks)
y_tick_range = img_grid.shape[0]
y_ticks = np.arange(0, y_tick_range+1e-9, step=y_tick_range/num_y_ticks)
plt.xticks(x_ticks, x_ticklabels)
plt.yticks(y_ticks, y_ticklabels)
plt.tight_layout()
plt.savefig('collage.pdf')
