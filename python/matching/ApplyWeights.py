import numpy as np

def ApplyWeights(w,p):
    S = np.shape(w)[0]
    Q = np.shape(p)[1]
    z = np.zeros((S,Q))
    if Q < S:
        p = p.transpose()
        copies = np.zeros((1,S))
        for q in range(Q):
            z[:,q] = np.sum(np.square((w-p[q + copies,:])),1)
    else:
        w = w.transpose()
        copies = np.zeros((1,Q))
        for i in range(S):
            z[i,:] = np.sum(np.square((np.transpose(np.tile(np.transpose(w[:,i]),(Q,1)))-p)),0)
    return np.sqrt(z)