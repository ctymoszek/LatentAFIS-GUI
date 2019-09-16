import math
from math import sin
from math import cos
import numpy as np
from skimage import transform as tf

def OF_SimilarityEstimation(result, template_q, template_t, options):
    result['ori_q'] = []
    result['ori_t'] = []
    
    if (not 'LSS_T_minu_index1' in result) or (not 'LSS_T_minu_index2' in result):
        return result
    if len(result['LSS_T_minu_index1'])<1 or len(result['LSS_T_minu_index2']) < 1:
        return result
    
    minu_q = template_q.minutiae[result['LSS_T_minu_index1']]
    minu_t = template_t.minutiae[result['LSS_T_minu_index2']]
    
    blk_size = 16
    if len(result['LSS_T_minu_index1']) < 3:
        #get dx, dy, and drot
        cos_theta = 0
        sin_theta = 0
        minu_shape = np.shape(minu_q)
        N_q = minu_shape[0]
        for i in range(N_q):
            cos_theta = cos_theta + cos(minu_t[i,2] - minu_q[i,2])
            sin_theta = sin_theta + sin(minu_t[i,2] - minu_q[i,2])
        rot = math.atan2(sin_theta, cos_theta)
        
        cos_theta = cos(rot)
        sin_theta = sin(rot)
        minu_t_rot = minu_t
        dx = 0
        dy = 0
        for i in range(N_q):
            x = minu_t[i,0]*cos_theta - minu_t[i,1]*sin_theta
            y = minu_t[i,0]*sin_theta + minu_t[i,1]*cos_theta
            minu_t_rot[i,0] = x
            minu_t_rot[i,1] = y
            
            dx = dx + minu_q[i,0] - x
            dy = dy + minu_q[i,1] - y
        minu_shape = np.shape(minu_t)
        N_t = minu_shape[0]
        dy = dy / N_t
        dx = dx / N_t
        
        tmp = np.where(template_t.oimg > -10)
        ind = np.unravel_index(tmp, (template_t.blkH, template_t.blkW))
        Y = ind[:,0]
        X = ind[0,:]
        X_r = X*cos_theta - Y*sin_theta + dx
        Y_r = X*sin_theta + Y*cos_theta + dx
    else:
        t = tf.SimilarityTransform()
        t.estimate(minu_t[:,0:2],minu_q[:,0:2])
        rot = t.rotation
        coords = np.where(template_t.oimg > -10)
        X = coords[0]
        Y = coords[1]
        t_scaled = tf.SimilarityTransform(scale = t.scale, rotation = t.rotation,
                                          translation = t.translation/blk_size)
        coords_t = t_scaled(np.transpose([X,Y]))
        X_r = coords_t[:,0]
        Y_r = coords_t[:,1]
        
    X_r = np.round(X_r).astype(int)
    Y_r = np.round(Y_r).astype(int)
    oimg_t = np.transpose(template_t.oimg)
    ori_t = np.add(oimg_t[oimg_t > -10], rot)
    shape_q = np.shape(template_q.oimg)
    h_q = shape_q[0]
    w_q = shape_q[1]
    ind_valid_mask = np.where(X_r < h_q,1,0)&np.where(X_r >= 0,1,0)&np.where(Y_r < w_q,1,0)&np.where(Y_r >= 0,1,0)
    ind_valid = np.where(ind_valid_mask)
    X_r = X_r[ind_valid]
    Y_r = Y_r[ind_valid]
    ori_t = ori_t[ind_valid]
    
    shape_t = np.shape(template_t.oimg)
    ind_t = np.ravel_multi_index([X,Y],shape_t)
    ind_q = np.ravel_multi_index([X_r,Y_r],shape_q)
    
    ori_q = np.reshape(np.transpose(template_q.oimg),(h_q*w_q))
    ori_q = ori_q[ind_q] - rot
    ind2 = np.where(ori_q > -10)
    
    ori_q = ori_q[ind2]
    ori_t = ori_t[ind2]
    result['ori_q'] = ori_q
    result['ori_t'] = ori_t
    
    return result