import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

filepath = 'clip.pdf'

def sampleFunction(f, start, end, numSamples):
    xs = []
    ys = []
    for i in range(numSamples):
        x = (end-start) * (i * 1.0 / numSamples) + start
        xs.append(x)
        ys.append(f(x))
    return xs, ys

startX = -8.0
endX = 8.0
numSamples = 10000

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

def clip_func(x):
    if abs(x) <= 5.12:
        return x
    return 5.12 / x

def clip_deriv_func(x):
    if abs(x) <= 5.12:
        return 1
    return -5.12 / (x * x)

# Pick the function to show
f = clip_func
#f = clip_deriv_func

xs, ys = sampleFunction(f, startX, endX, numSamples)

print(xs)
print(ys)

plt.plot(xs, ys, 'k')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Draw plot 1 with uniform lines accross a value
plt.tight_layout()
#plt.show()

plt.savefig(filepath)

plt.close('all')
