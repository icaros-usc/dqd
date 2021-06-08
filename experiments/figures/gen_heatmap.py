# Based on pyribs visualize.py
# Visualizes a custom heatmap of an archive.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable

domain_name = 'lin_proj'
#domain_name = 'arm'
#domain_name = 'lsi_clip'

algorithm_name = 'cma_mega_adam'
archive_filename = f"../{domain_name}/logs/{algorithm_name}/trial_0/archive_10000.pkl"

if domain_name == 'lin_proj':
    lb_val = (-5.12 / 2) * 1000.0
    ub_val = (5.12 / 2) * 1000.0
    x_dim, y_dim = (100, 100)
    vmin = 0.0
    vmax = 100.0
elif domain_name == 'arm':
    lb_val = -1000.0                # lin proj domain bounds
    ub_val = 1000.0                 # lin proj domain bounds
    x_dim, y_dim = (100, 100)
    vmin = 99.999
    #vmin = 0.0
    vmax = 100.0
elif domain_name == 'lsi_clip':
    lb_val = 0.0
    ub_val = 6.0
    x_dim, y_dim = (200, 200)
    vmin = 0.0
    vmax = 100.0

transpose_bcs = False
square = False

def _retrieve_cmap(cmap):
    """Retrieves colormap from matplotlib."""
    if isinstance(cmap, str):
        return matplotlib.cm.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap

# Try getting the colormap early in case it fails.
cmap = 'magma'
cmap = _retrieve_cmap(cmap)

# Archive bounds
lower_bounds = (lb_val, lb_val)
upper_bounds = (ub_val, ub_val)
x_bounds = np.linspace(lower_bounds[0], upper_bounds[0], x_dim + 1)
y_bounds = np.linspace(lower_bounds[1], upper_bounds[1], y_dim + 1)

# Color for each cell in the heatmap.
archive_data = pd.read_pickle(archive_filename)
colors = np.full((y_dim, x_dim), np.nan)
for row in archive_data.itertuples():
    obj = np.sqrt(row.objective / 100.0) * 100.0
    colors[row.index_1, row.index_0] = obj
objective_values = archive_data["objective"]

if transpose_bcs:
    # Since the archive is 2D, transpose by swapping the x and y boundaries
    # and by flipping the bounds (the bounds are arrays of length 2).
    x_bounds, y_bounds = y_bounds, x_bounds
    lower_bounds = np.flip(lower_bounds)
    upper_bounds = np.flip(upper_bounds)
    colors = colors.T

# Initialize the axis.
ax = plt.gca()
ax.set_xlim(lower_bounds[0], upper_bounds[0])
ax.set_ylim(lower_bounds[1], upper_bounds[1])

if square:
    ax.set_aspect("equal")

# Create the plot.
pcm_kwargs = {}
vmin = np.min(objective_values) if vmin is None else vmin
vmax = np.max(objective_values) if vmax is None else vmax
t = ax.pcolormesh(x_bounds,
                  y_bounds,
                  colors,
                  cmap=cmap,
                  vmin=vmin,
                  vmax=vmax,
                  norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                  **pcm_kwargs)

# Create the colorbar.
ax.figure.colorbar(t, ax=ax, pad=0.1)
#plt.show()
plt.savefig('heatmap.png')
