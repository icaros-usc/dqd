import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

data = pd.read_csv('summary.csv')

y_label = "Coverage"

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

sns.set(font_scale=3)
with sns.axes_style("white"):
    sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})
    sns.set_palette("colorblind")
    
    #fig, ax = plt.subplots()    

    # Plot the responses for different events and regions
    sns_plot = sns.lineplot(x="Iteration", 
                            y=y_label, 
                            hue="Algorithm",
                            data=data, 
                            legend=False, 
                            palette=palette,
                           )
    plt.xticks([0, 5000, 10000])
    plt.yticks([0, 100])
    plt.xlabel("Iterations")
    plt.ylabel(y_label)

    legend = plt.legend(loc='best',frameon=False, prop={'size': 25})
    legend.set_bbox_to_anchor((0.48, 0.65))
    
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.tight_layout()
    #plt.show()
    sns_plot.figure.savefig("coverage.pdf")
