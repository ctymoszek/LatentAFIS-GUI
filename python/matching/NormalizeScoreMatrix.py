import numpy as np

def NormalizeScoreMatrix(simi):
    simi_shape = np.shape(simi)
    num_minu1 = simi_shape[0]
    num_minu2 = simi_shape[1]
    sum1 = np.sum(simi,1)
    sum1 = np.transpose(np.tile(sum1,[num_minu2,1]))
    sum2 = np.sum(simi,0)
    sum2 = np.tile(sum2,[num_minu1, 1])
    
    norm_simi = simi/(sum1 + sum2 - simi + np.spacing(1))
    
    return norm_simi