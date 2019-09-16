import numpy as np
from math import isnan

def CalScores(result, method, score_file=None):
    num_1 = 1
    num_2 = 1
    
    score = np.zeros((num_1,num_2))
    if method == 1:
        for i in range(num_1):
            for j in range(num_2):
                if 'LSS_T_simi' in result and len(result['LSS_T_simi']) > 0 :
                    of_simi = abs(np.cos(np.subtract(result['ori_q'],result['ori_t']))) 
                    of_simi = np.mean(of_simi)
                    if isnan(of_simi):
                        of_simi = 0
                    score[i,j] = mean2(result['local_compat'] * sum(result['LSS_T_simi'] \
                                         * of_simi * sum(result['minu_str_simi'])))
    elif method == 2:
        for i in range(num_1):
            for j in range(num_2):
                R = result[i,j]
                if 'patch_simi' in R:
                    score[i,j] = sum(R['patch_simi'])
    elif method == 3:
        for i in range(num_1):
            for j in range(num_2):
                R = result[i,j]
                if 'LSS_T_simi' in R:
                    score[i,j] = sum(R['LSS_T_simi'])
    if score_file != None:
        np.savetxt(score_file, score)
    return score

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y