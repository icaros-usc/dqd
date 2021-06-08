import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

data = pd.read_csv('cdf.csv')

y_label = "Threshold Percentage"

plt.figure(figsize = (12,12))

# Color mapping for algorithms
palette ={
    "CMA-MEGA": "C0", 
    "CMA-MEGA (Adam)": "C1", 
    "OMG-MEGA": "C2", 
    "OG-MAP-Elites": "C3",
    "CMA-ME": "C4",
    "MAP-Elites": "C5",
    "MAP-Elites (line)": "C6",
}

sns.set(font_scale=4)
with sns.axes_style("white"):
    sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})
    sns.set_palette("colorblind")
    
    # Plot the responses for different events and regions
    sns_plot = sns.lineplot(x="Objective", 
                            y=y_label, 
                            linewidth=3.0,
                            hue="Algorithm",
                            data=data, 
                            legend=False, 
                            palette=palette,
                           )
    plt.xticks([0, 50, 100])
    plt.yticks([0, 100])
    plt.xlabel("Objective")
    plt.ylabel(y_label)

    legend = plt.legend(loc='center', frameon=False, prop={'size': 26})
    legend.set_bbox_to_anchor((0.35, 0.45))
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.tight_layout()
    #plt.show()
    sns_plot.figure.savefig("cdf.pdf")
