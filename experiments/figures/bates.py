import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import functools

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

filepath = 'bates.pdf'

sns.set(style="white", palette="muted", color_codes=True)


def sampleFunction(f, start, end, numSamples):
    xs = []
    ys = []
    for i in range(numSamples):
        x = (end-start) * (i * 1.0 / numSamples) + start
        xs.append(x)
        ys.append(f(x))
    return xs, ys

startX = 0.0
endX = 1.0
numSamples = 10000

def fact(n):
    if n == 0:
        return 1
    return functools.reduce(lambda a, b: a*b, range(1, n+1))

@functools.lru_cache(maxsize=None)
def binom(n, k):
    if n == 0 or k == 0 or n == k:
        return 1
    return binom(n-1, k) + binom(n-1, k-1)
for i in range(10):
    print([binom(i,k) for k in range(i+1)])

def generateBatesPDF(n):
    def f(x):
        coeff = n / (2.0*fact(n-1))

        summation = 0
        lead = 1.0
        for k in range(n+1):
            v1 = binom(n, k)
            v2 = (n*x-k) ** (n-1)
            nx = n * x
            s = 0
            if nx < k:
                s = -1
            else:
                s = 1
            cur = lead * v1 * v2 * s
            summation += cur
            lead *= -1
        return coeff * summation

    return f

data = {}
column_order = []
for n in [1, 2, 5, 20]:
    xs, ys = sampleFunction(generateBatesPDF(n), startX, endX, numSamples)
    label = 'n={}'.format(n)
    column_order.append(label)
    data[label] = ys

xs, _ = sampleFunction(lambda x: 1, startX, endX, numSamples)
xlabels = {x:xs[x] for x in range(numSamples)}

data = pd.DataFrame(data)
data = data.rename(index=xlabels)
data = data[column_order]
print(data)

#sns.set(font_scale=3)
with sns.axes_style("white"):
    sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})

    # Draw plot 1 with uniform lines accross a value
    p = sns.color_palette(palette=['Black'] * 4)
    p1 = sns.lineplot(data=data, palette=p, linewidth=2.5)

    #sns.despine(left=True, right=True, top=True)

    p1.axis('off')

    plt.legend(prop={'size':20})
    plt.tight_layout()
    #plt.show()
    p1.figure.savefig(filepath, bbox_inches='tight')

    plt.close('all')
