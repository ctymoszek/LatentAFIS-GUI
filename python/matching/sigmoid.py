import numpy as np

def sigmoid(v, mu, tau):
    z = 1 / (1 + np.exp(-tau * (v - mu)))
    return z