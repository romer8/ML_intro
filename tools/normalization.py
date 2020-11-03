import numpy as np


def normalizeData(x):
    x_normed = (x - x.min(0)) / x.ptp(0)
    return x_normed

def giveMaxValues(X):
    maxvalues = X.max(0)
    return maxvalues
def giveMinValues(X):
    minvalues = X.min(0)
    return minvalues
