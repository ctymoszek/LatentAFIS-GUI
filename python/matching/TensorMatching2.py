import numpy as np
from numpy import pi, arctan2
import math
from dphi import dphi
from ApplyWeights import ApplyWeights

def TensorMatching2(simi, corr, minu_temp1, minu_temp2):
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
    ind = DPhi2 < -pi
    DPhi2[ind] = DPhi2[ind] + 2*pi
    ind = DPhi2 > pi
    DPhi2[ind] = DPhi2[ind] - 2*pi

    ori_diff = abs(DPhi1 - DPhi2)

    ind = ori_diff > pi
    ori_diff[ind] = 2*pi - ori_diff[ind]

    rho = np.zeros((N,N,N))
    L_x = np.empty((3))
    L_y = np.empty((3))
    r_x = np.empty((3))
    r_y = np.empty((3))
    for p in range(N):
        for t in range(p + 1, N):

            #d1
            dis1 = D1[p,t]
            dis2 = D2[p,t]
            d1 = abs(dis1 - dis2)
            if d1 > 30:
                continue

            #d2
            d2 = ori_diff[p,t]

            #d3
            dr1 = dphi(minu_temp1[corr[0][t]][2],
                       -arctan2(minu_temp1[corr[0][p]][1]-minu_temp1[corr[0][t]][1],
                                minu_temp1[corr[0][p]][0]-minu_temp1[corr[0][t]][0]))
            dr2 = dphi(minu_temp2[corr[1][t]][2],
                       -arctan2(minu_temp2[corr[1][p]][1]-minu_temp2[corr[1][t]][1],
                                minu_temp2[corr[1][p]][0]-minu_temp2[corr[1][t]][0]))
            d3 = abs(dphi(dr1, dr2))

            #d4
            dr1 = dphi(minu_temp1[corr[0][p]][2],
                       -arctan2(minu_temp1[corr[0][t]][1]-minu_temp1[corr[0][p]][1],
                                minu_temp1[corr[0][t]][0]-minu_temp1[corr[0][p]][0]))
            dr2 = dphi(minu_temp2[corr[1][p]][2],
                       -arctan2(minu_temp2[corr[1][t]][1]-minu_temp2[corr[1][p]][1],
                                minu_temp2[corr[1][t]][0]-minu_temp2[corr[1][p]][0]))
            d4 = abs(dphi(dr1, dr2))

            if d1 > 30 or d2 > pi/4 or d3 > pi/6 or d4 > pi/6:
                continue
            for k in range(t + 1, N):
                #d1
                dis1 = D1[t,k]
                dis2 = D2[t,k]
                d1 = abs(dis1 - dis2)

                #d2
                d2 = ori_diff[k,t]

                #d3
                dr1 = dphi(minu_temp1[corr[0][t]][2],
                           -arctan2(minu_temp1[corr[0][k]][1]-minu_temp1[corr[0][t]][1],
                                    minu_temp1[corr[0][k]][0]-minu_temp1[corr[0][t]][0]))
                dr2 = dphi(minu_temp2[corr[1][t]][2],
                                -arctan2(minu_temp2[corr[1][k]][1]-minu_temp2[corr[1][t]][1],
                                            minu_temp2[corr[1][k]][0]-minu_temp2[corr[1][t]][0]))
                d3 = abs(dphi(dr1, dr2))

                #d4
                dr1 = dphi(minu_temp1[corr[0][k]][2],
                           -arctan2(minu_temp1[corr[0][t]][1]-minu_temp1[corr[0][k]][1],
                                    minu_temp1[corr[0][t]][0]-minu_temp1[corr[0][k]][0]))
                dr2 = dphi(minu_temp2[corr[1][k]][2],
                           -arctan2(minu_temp2[corr[1][t]][1]-minu_temp2[corr[1][k]][1],
                                    minu_temp2[corr[1][t]][0]-minu_temp2[corr[1][k]][0]))
                d4 = abs(dphi(dr1, dr2))

                if d1 > 30 or d2 > pi/4 or d3 > pi/6 or d4 > pi/6:
                    continue

                #d1
                dis1 = D1[p,k]
                dis2 = D2[p,k]
                d1 = abs(dis1 - dis2)

                #d2
                d2 = ori_diff[k,p]

                #d3
                dr1 = dphi(minu_temp1[corr[0][p]][2],
                           -arctan2(minu_temp1[corr[0][k]][1]-minu_temp1[corr[0][p]][1],
                                    minu_temp1[corr[0][k]][0]-minu_temp1[corr[0][p]][0]))
                dr2 = dphi(minu_temp2[corr[1][p]][2],
                           -arctan2(minu_temp2[corr[1][k]][1]-minu_temp2[corr[1][p]][1],
                                    minu_temp2[corr[1][k]][0]-minu_temp2[corr[1][p]][0]))
                d3 = abs(dphi(dr1, dr2))

                #d4
                dr1 = dphi(minu_temp1[corr[0][k]][2],
                           -arctan2(minu_temp1[corr[0][p]][1]-minu_temp1[corr[0][k]][1],
                                    minu_temp1[corr[0][p]][0]-minu_temp1[corr[0][k]][0]))
                dr2 = dphi(minu_temp2[corr[1][k]][2],
                           -arctan2(minu_temp2[corr[1][p]][1]-minu_temp2[corr[1][k]][1],
                                    minu_temp2[corr[1][p]][0]-minu_temp2[corr[1][k]][0]))
                d4 = abs(dphi(dr1, dr2))

                if d1 > 30 or d2 > pi/4 or d3 > pi/6 or d4 > pi/6:
                    continue

                L_x[0] = minu_temp1[corr[0][p]][0]
                L_x[1] = minu_temp1[corr[0][t]][0]
                L_x[2] = minu_temp1[corr[0][k]][0]
                L_y[0] = minu_temp1[corr[0][p]][1]
                L_y[1] = minu_temp1[corr[0][t]][1]
                L_y[2] = minu_temp1[corr[0][k]][1]
                L_x = np.insert(L_x, 0, L_x[len(L_x)-1])
                L_x = np.append(L_x, L_x[1])
                L_y = np.insert(L_y, 0, L_y[len(L_y)-1])
                L_y = np.append(L_y, L_y[1])

                r_x[0] = minu_temp2[corr[1][p]][0]
                r_x[1] = minu_temp2[corr[1][t]][0]
                r_x[2] = minu_temp2[corr[1][k]][0]
                r_y[0] = minu_temp2[corr[1][p]][1]
                r_y[1] = minu_temp2[corr[1][t]][1]
                r_y[2] = minu_temp2[corr[1][k]][1]
                r_x = np.insert(r_x, 0, r_x[len(r_x)-1])
                r_x = np.append(r_x, r_x[1])
                r_y = np.insert(r_y, 0, r_y[len(r_y)-1])
                r_y = np.append(r_y, r_y[1])

                ind = [p, t, k]
                flag = 1
                for i in range(3):
                    L_y_d1 = L_y[i+2] - L_y[i+1]
                    L_x_d1 = L_x[i+2] - L_x[i+1]
                    L_d1 = L_y_d1*L_y_d1 + L_x_d1*L_x_d1
                    L_d1 = math.sqrt(L_d1) + np.spacing(1)
                    L_y_d1 = L_y_d1/L_d1
                    L_x_d1 = L_x_d1/L_d1

                    L_y_d2 = L_y[i] - L_y[i+1]
                    L_x_d2 = L_x[i] - L_x[i+1]
                    L_d2 = L_y_d2*L_y_d2 + L_x_d2*L_x_d2
                    L_d2 = math.sqrt(L_d2) + np.spacing(1)
                    L_y_d2 = L_y_d2/L_d2
                    L_x_d2 = L_x_d2/L_d2

                    L_angle1 = dphi(arctan2(L_y_d1, L_x_d1),
                                    arctan2(L_y_d2, L_x_d2))
                    L_bisect = -arctan2(L_y_d2 + L_y_d1, L_x_d2 + L_x_d1)
                    L_minu_ori = minu_temp1[corr[0][ind[i]]][2]
                    L_angle2 = dphi(L_bisect, L_minu_ori)

                    r_y_d1 = r_y[i+2] - r_y[i+1]
                    r_x_d1 = r_x[i+2] - r_x[i+1]
                    r_d1 = r_y_d1*r_y_d1 + r_x_d1*r_x_d1
                    r_d1 = math.sqrt(r_d1) + np.spacing(1)
                    r_y_d1 = r_y_d1/r_d1
                    r_x_d1 = r_x_d1/r_d1

                    r_y_d2 = r_y[i] - r_y[i+1]
                    r_x_d2 = r_x[i] - r_x[i+1]
                    r_d2 = r_y_d2*r_y_d2 + r_x_d2*r_x_d2
                    r_d2 = math.sqrt(r_d2) + np.spacing(1)
                    r_y_d2 = r_y_d2/r_d2
                    r_x_d2 = r_x_d2/r_d2

                    r_angle1 = dphi(arctan2(r_y_d1, r_x_d1),
                                    arctan2(r_y_d2, r_x_d2))
                    r_bisect = -arctan2(r_y_d2 + r_y_d1, r_x_d2 + r_x_d1)
                    r_minu_ori = minu_temp2[corr[1][ind[i]]][2]
                    r_angle2 = dphi(r_bisect, r_minu_ori)

                    angle_diff1 = abs(L_angle1 - r_angle1)
                    if angle_diff1 > pi:
                        angle_diff1 = 2*pi - angle_diff1
                    if angle_diff1 > pi/6:
                        flag = 0
                        break

                    angle_diff2 = abs(L_angle2 - r_angle2)
                    if angle_diff2 > pi:
                        angle_diff2 = 2*pi - angle_diff2
                    if angle_diff2 > pi/6:
                        flag = 0
                        break
                if flag == 0:
                    continue
                s = 1
                rho[k][t][p] = s
                rho[k][p][t] = s
                rho[t][k][p] = s
                rho[t][p][k] = s
                rho[p][t][k] = s
                rho[p][k][t] = s
    wR = 0.5
    S0 = np.ones((len(S), 1))
    S0 = S0 / sum(S0)

    for n in range(20):
        SS = np.zeros(np.shape(S))
        for i in range(N):
            s = 0
            for j in range(N):
                for k in range(N):
                    s = s + rho[i][j][k]*S[j]*S[k]
            SS[i] = s
        S = SS / (sum(SS) + np.spacing(1))

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
        mflag[ind[i]] = 1
        flag1[corr[0][ind[i]]] = 1
        flag2[corr[1][ind[i]]] = 1

        selected_ind.append(ind[i])
    local_compat = np.empty((len(selected_ind),len(selected_ind),len(selected_ind)))
    for i in range(len(selected_ind)):
        for j in range(len(selected_ind)):
            for k in range(len(selected_ind)):
                local_compat[i][j][k] = rho[selected_ind[i]][selected_ind[j]][selected_ind[k]]
    corr[0] = corr[0][np.nonzero(mflag)[0]]
    corr[1] = corr[1][np.nonzero(mflag)[0]]

    return corr, local_compat
