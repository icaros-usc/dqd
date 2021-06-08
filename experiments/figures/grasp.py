import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

filepath = 'grasp.pdf'

def visualize(ax, solution, link_lengths):

    lim = 1.05 * np.sum(link_lengths)
    
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Plot each link and joint
    pos = np.array([0,0])
    cum_thetas = np.cumsum(solution)
    
    for link_length, cum_theta in zip(link_lengths, cum_thetas):
        
        next_pos = pos + link_length * np.array(
                [np.cos(cum_theta), np.sin(cum_theta)]
            )
        ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], '-ko', ms=3)
        pos = next_pos

    ax.plot(0, 0, "ro", ms=6)

    #final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"

    #ax.plot(pos[0], pos[1], "go", ms=6, label=final_label)
    ax.plot(pos[0], pos[1], "go", ms=6)
    #ax.legend()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

n = 7
links = np.array([1.0,] * n)


#angles = np.array([0.65,] * n)
#angles = np.array([2.65, 0.75, 0.3, 0.6, -0.3, -0.23, -0.5])
#visualize(angles, links)


num_points = 19
lo_bound = -0.7
hi_bound = 0.7
delta = hi_bound - lo_bound
for i in range(num_points):
    a = delta * (i / (num_points - 1)) + lo_bound
    print(a)
    
    angles = np.array([a,] * n)
    visualize(ax, angles, links)

plt.savefig(filepath, bbox_inches='tight')
plt.close('all')
