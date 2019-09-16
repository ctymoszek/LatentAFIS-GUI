import numpy as np
from numpy import pi
from sigmoid import sigmoid
from ApplyWeights import ApplyWeights

def LSS_R_Fast2(simi, corr, minu_temp1, minu_temp2):
    minu_simi_shape = np.shape(simi)
    num_minu1 = minu_simi_shape[0]
    num_minu2 = minu_simi_shape[1]
    ind = np.ravel_multi_index((corr[0],corr[1]),np.shape(simi), order='F')
    S = np.reshape(simi,(num_minu1 * num_minu2, 1), order='F')[ind]

    nR = 5;

    D1 = ApplyWeights(minu_temp1[corr[0],0:2],np.transpose(minu_temp1[corr[0],0:2]))
    D2 = ApplyWeights(minu_temp2[corr[1],0:2],np.transpose(minu_temp2[corr[1],0:2]))

    corr_shape = np.shape(corr)
    N = corr_shape[1]

    ori = np.tile(minu_temp1[corr[0],2],[N,1])
    DPhi1 = ori - ori.transpose()
    ind = DPhi1 < -pi
    DPhi1[ind] = DPhi1[ind] + 2*pi
    ind = DPhi1 > pi
    DPhi1[ind] = DPhi1[ind] - 2*pi

    ori = np.tile(minu_temp2[corr[1],2],[N,1])
    DPhi2 = ori - ori.transpose()
    ind = DPhi2 <= -pi
    DPhi2[ind] = DPhi2[ind] + 2*pi
    ind = DPhi2 > pi
    DPhi2[ind] = DPhi2[ind] - 2*pi

    ori_diff = abs(DPhi1 - DPhi2)

    ind = ori_diff > pi
    ori_diff[ind] = 2*pi - ori_diff[ind]

    DX1 = np.subtract(np.tile(np.transpose([minu_temp1[corr[0],0]]),[1,N]),
                      np.tile(minu_temp1[corr[0],0],[N,1]))
    DY1 = np.subtract(np.tile(np.transpose([minu_temp1[corr[0],1]]),[1,N]),
                      np.tile(minu_temp1[corr[0],1],[N,1]))
    line_ori1 = - np.arctan2(DY1,DX1)
    dir_line_diff1 = np.subtract(np.tile(np.transpose([minu_temp1[corr[0],2]]),[1,N]), line_ori1)
    ind = dir_line_diff1 <= -pi
    dir_line_diff1[ind] = dir_line_diff1[ind] + 2*pi
    ind = dir_line_diff1 > pi
    dir_line_diff1[ind] = dir_line_diff1[ind] - 2*pi

    DX2 = np.subtract(np.tile(np.transpose([minu_temp2[corr[1],0]]),[1,N]),
                      np.tile(minu_temp2[corr[1],0],[N,1]))
    DY2 = np.subtract(np.tile(np.transpose([minu_temp2[corr[1],1]]),[1,N]),
                      np.tile(minu_temp2[corr[1],1],[N,1]))
    line_ori2 = - np.arctan2(DY2,DX2)
    dir_line_diff2 = np.subtract(np.tile(np.transpose([minu_temp2[corr[1],2]]),[1,N]), line_ori2)
    ind = dir_line_diff2 <= -pi
    dir_line_diff2[ind] = dir_line_diff2[ind] + 2*pi
    ind = dir_line_diff2 > pi
    dir_line_diff2[ind] = dir_line_diff2[ind] - 2*pi

    dir_line_diff = abs(dir_line_diff1 - dir_line_diff2)
    ind = dir_line_diff > pi
    dir_line_diff[ind] = 2*pi - dir_line_diff[ind]

    d1 = abs(D1 - D2)
    d2 = ori_diff
    d3 = dir_line_diff
    d4 = d3.transpose()
    rho = sigmoid(d1, 20, -0.2) * sigmoid(d2, pi/4, -10) \
             * sigmoid(d3, pi/6, -10) * sigmoid(d4, pi/6, -10)
    rho_mask = np.where(D1==0,1,0)|np.where(d1>40,1,0)|np.where(d2>pi/4,1,0)|np.where(d3>pi/6,1,0)|np.where(d4>pi/6,1,0)
    rho = np.where(rho_mask == 0, rho, 0)

    wR = 0.5
    S0 = np.ones((len(S), 1))
    S0 = S0 / sum(S0)

    for n in range(20):
        S = np.dot(rho, S)
        S = S / (sum(S) + np.spacing(1))

    #sort
    epsilon = S / (sum(S) + np.spacing(1))
    ind = np.transpose(np.argsort(epsilon, axis=0)[::-1])[0]

    #get minutiae correspondence based on greedy rules
    flag1 = np.zeros((max(corr[0])+1,1))
    flag2 = np.zeros((max(corr[1])+1,1))

    mflag = np.zeros((N,1))
    selected_ind = []
    for i in range(N):
        if epsilon[ind[i]] < 0.001:
            break
        if flag1[corr[0][ind[i]]] == 1 or flag2[corr[1][ind[i]]] == 1:
            continue

        s = rho[ind[i], selected_ind]
        if len(np.where(s == 0)[0]) > 0 and i != 0:
            continue
        mflag[ind[i]] = 1
        flag1[corr[0][ind[i]]] = 1
        flag2[corr[1][ind[i]]] = 1

        selected_ind.append(ind[i])
    local_compat = np.empty((len(selected_ind),len(selected_ind)))
    for i in range(len(selected_ind)):
        for j in range(len(selected_ind)):
            local_compat[i][j] = rho[selected_ind[i]][selected_ind[j]]
    corr[0] = corr[0][np.nonzero(mflag)[0]]
    corr[1] = corr[1][np.nonzero(mflag)[0]]

    return corr, local_compat
