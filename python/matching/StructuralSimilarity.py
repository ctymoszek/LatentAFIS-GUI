import numpy as np
from numpy import pi, sqrt
from math import atan2, sin, cos
from GetMinutiaeConvexHull import GetMinutiaeConvexHull
import matplotlib.path as mpltPath
from skimage import transform as tf

def StructuralSimilarity(corr, minutiae1, minutiae2, Q_1, Q_2):
    minu_q = minutiae1[corr[0],:]
    minu_t = minutiae2[corr[1],:]

    if np.shape(corr)[1] <= 2:
        minu_str_simi = 0
        return

    t = tf.SimilarityTransform()
    t.estimate(minu_q[:,0:2],minu_t[:,0:2])
    rot = t.rotation
    #calculate unmatched minutiae
    #transform the query minutiae
    minutiae_aligned = np.copy(minutiae1)
    minutiae_aligned[:,0:2] = t(np.transpose([np.transpose(minutiae1[:,0:1])[0],np.transpose(minutiae1[:,1:2])[0]]))
    minutiae_aligned[:,2] = minutiae_aligned[:,2] - rot
    minutiae_aligned[:,2] = np.mod(minutiae_aligned[:,2],2*pi)

    #compute convex hull of the two minutiae sets
    R = 10
    xv_T, yv_T = GetMinutiaeConvexHull(minutiae2, R)
    xv_Q, yv_Q = GetMinutiaeConvexHull(minutiae_aligned, R)

    polygon = [xv_Q, yv_Q]
    path = mpltPath.Path(np.transpose(polygon))
    in_2 = path.contains_points(np.transpose([minutiae2[:,0],minutiae2[:,1]]))
    in_2 = np.nonzero(in_2)
    polygon = [xv_T, yv_T]
    path = mpltPath.Path(np.transpose(polygon))
    in_1 = path.contains_points(np.transpose([minutiae_aligned[:,0],minutiae_aligned[:,1]]))
    in_1 = np.nonzero(in_1)

    #coordinates of the matched minutiae
    x_1 = minutiae1[corr[0],0]
    y_1 = minutiae1[corr[0],1]
    ori_1 = minutiae1[corr[0],2]
    tQ_1 = Q_1[corr[0]]

    x_2 = minutiae2[corr[1],0]
    y_2 = minutiae2[corr[1],1]
    ori_2 = minutiae2[corr[1],2]
    tQ_2 = Q_2[corr[1]]

    matched_N = np.shape(corr)[1]
    minu_str_simi = np.zeros(matched_N)

    for i in range(matched_N):
        len_1 = sqrt(np.square(np.subtract(minutiae1[in_1,0],x_1[i])) + \
                     np.square(np.subtract(minutiae1[in_1,1],y_1[i])))
        len_2 = sqrt(np.square(np.subtract(minutiae2[in_2,0],x_2[i])) + \
                     np.square(np.subtract(minutiae2[in_2,1],y_2[i])))

        ind_1 = np.where(len_1 < 80)[1]
        ind_2 = np.where(len_2 < 80)[1]

        ind_1 = np.setdiff1d(in_1[0][ind_1], corr[0])
        ind_2 = np.setdiff1d(in_2[0][ind_2], corr[1])

        penalty_1 = sum(Q_1[ind_1])
        penalty_2 = sum(Q_2[ind_2])

        matchedlen_1 = sqrt(np.square(x_1 - x_1[i]) + np.square(y_1 - y_1[i]))
        matchedlen_2 = sqrt(np.square(x_2 - x_2[i]) + np.square(y_2 - y_2[i]))

        local_ind_mask = np.where(matchedlen_1 < 80,1,0)|np.where(matchedlen_2 < 80,1,0)
        local_ind = np.where(local_ind_mask)[0]

        s = 0
        w = 0
        for j in range(len(local_ind)):
            k = local_ind[j]
            theta_1 = -ori_1[i] - atan2(y_1[i] - y_1[k], x_1[i] - x_1[k])
            theta_2 = -ori_2[i] - atan2(y_2[i] - y_2[k], x_2[i] - x_2[k])

            tx_1 = matchedlen_1[k] * cos(theta_1)
            ty_1 = matchedlen_1[k] * sin(theta_1)

            tx_2 = matchedlen_2[k] * cos(theta_2)
            ty_2 = matchedlen_2[k] * sin(theta_2)

            d_1 = sqrt(np.square(tx_1 - tx_2) + np.square(ty_1 - ty_2))

            theta_1 = -ori_1[k] - atan2(y_1[i] - y_1[k], x_1[i] - x_1[k])
            theta_2 = -ori_2[k] - atan2(y_2[i] - y_2[k], x_2[i] - x_2[k])

            tx_1 = matchedlen_1[k] * cos(theta_1)
            ty_1 = matchedlen_1[k] * sin(theta_1)

            tx_2 = matchedlen_2[k] * cos(theta_2)
            ty_2 = matchedlen_2[k] * sin(theta_2)

            d_2 = sqrt(np.square(tx_1 - tx_2) + np.square(ty_1 - ty_2))

            w = w + 2*(tQ_1[k] + tQ_2[k])
            if d_1 >= 30:
                s = s + 0
            else:
                s = s + (30 - d_1)/30 * (tQ_1[k] + tQ_2[k])

            if d_2 >= 30:
                s = s + 0
            else:
                s = s + (30 - d_2)/30 * (tQ_1[k] + tQ_2[k])
        minu_str_simi[i] = s / (w + penalty_1 + penalty_2)
    return minu_str_simi
