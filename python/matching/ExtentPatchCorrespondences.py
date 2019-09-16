from math import ceil, sin, cos, atan2, pi
import numpy as np

def ExtentPatchCorrespondences(result, template_Q, template_T, simi):
    #Q: Query latent print
    #T: Template rolled print
    result['ori_Q'] = []
    result['ori_T'] = []
    patch_simi = 0
    if not 'LSS_T_minu_index1' in result or not 'LSS_T_minu_index2' in result:
        return
    
    minu_Q = template_Q.minutiae[result['LSS_T_minu_index1'],:]
    minu_T = template_T.minutiae[result['LSS_T_minu_index2'],:]
    
    mask_shape = np.shape(template_T.mask)
    h = mask_shape[0]
    w = mask_shape[1]
    blk_size = 16
    blkH = ceil(h / blk_size)
    blkW = ceil(w / blk_size)
    minu_map = np.zeros((blkH,blkW))
    for i in range(np.shape(template_T.minutiae)[0]):
        y = template_T.minutiae[i,1]
        x = template_T.minutiae[i,0]
        y = round((y - blk_size/2) / blk_size) + 1
        x = round((x - blk_size/2) / blk_size) + 1
        minu_map[y,x] = i
    
    cos_theta = 0
    sin_theta = 0
    N = np.shape(minu_Q)[0]
    for i in range(N):
        cos_theta = cos_theta + cos(minu_Q[i,2] - minu_T[i,2])
        sin_theta = sin_theta + sin(minu_Q[i,2] - minu_T[i,2])
    rot = atan2(sin_theta, cos_theta)
    
    cos_theta = cos(rot)
    sin_theta = sin(rot)
    #commented out because not used
    #minu_T_rot = minu_T
    dx = 0
    dy = 0
    
    for i in range(N):
        x = minu_Q[i,0] * cos_theta - minu_Q[i,1] * sin_theta
        y = minu_Q[i,0] * sin_theta - minu_Q[i,1] * cos_theta
        dx = dx + minu_T[i,0] - x
        dy = dy + minu_T[i,1] - y
    dy = dy / N
    dx = dx / N
    
    all_minu_Q = template_Q.minutiae
    x = all_minu_Q[:,0] * cos_theta - all_minu_Q[:,1] * sin_theta + dx
    y = all_minu_Q[:,0] * sin_theta + all_minu_Q[:,1] * cos_theta + dy
    ori_Q = all_minu_Q[:,2] - rot
    
    x = round((x - blk_size/2)/blk_size) + 1
    y = round((y - blk_size/2)/blk_size) + 1
    all_minu_T = template_T.minutiae
    
    patch_simi = 0
    
    for i in range(len(y)):
        if y[i] < 1 or x[i] < 1 or y[i] > blkH or x[i] > blkW:
            continue
        if all_minu_Q[i,3] < 1.5:
            continue
        neast_ind = minu_map[y[i],x[i]]
        if neast_ind == 0:
            continue
        ori_T = all_minu_T[neast_ind,2]
        ori_diff = abs(ori_Q[i] - ori_T[i])
        if ori_diff > pi:
            ori_diff = 2*pi - ori_diff
        if ori_diff < pi/4:
            patch_simi = [patch_simi, simi[i,neast_ind]]
    return patch_simi