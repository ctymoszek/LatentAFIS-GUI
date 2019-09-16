import numpy as np
from scipy.spatial.distance import cdist
from NormalizeScoreMatrix import NormalizeScoreMatrix
from LSS_R_Fast2 import LSS_R_Fast2
from TensorMatching2 import TensorMatching2
from ExtentPatchCorrespondences import ExtentPatchCorrespondences

def PatchCorrespondences(template1, template2, options):
    num_minu1 = len(template1.minutiae)
    num_minu2 = len(template2.minutiae)

    result = {}
    result['num_minu1'] = num_minu1
    result['num_minu2'] = num_minu2

    minu_simi = np.zeros((num_minu1,num_minu2))

    result['minutiae1'] = template1.minutiae
    result['minutiae2'] = template2.minutiae

    result['patch_simi'] = []

    if num_minu1 < 3 or num_minu2 < 3:
        return
    if max(template1.minutiae[:,0]) > 1000 or max(template2.minutiae[:,0]) > 1000:
        result = []
    if 'method' in options and options['method'] == 'MCC':
        minu_simi = GetMCCSimilarity(template1,template2)
    elif 'method' in options and options['method'] == 'DNN':
        simi_parameter = options['simi_parameter']
        simi_parameter['patchType'] = options['patchType']
        minu_simi_ridge = GetDNNSimilarity(template1.des,template2.des,
                                           simi_parameter)
        minu_simi = minu_simi_ridge
    elif 'method' in options and options['method'] == 'PCA':
        minu_simi = 3 - cdist(template1.des_PCA,template2.des_PCA)

    norm_simi = NormalizeScoreMatrix(minu_simi)
    Q = template1.minutiae[:,4]
    norm_simi[Q < 1.5,:] = 0
    if 'num_match_init' in options:
        num_match_init = options['num_match_init']
    else:
        num_match_init = 120

    if num_match_init > (num_minu1 * num_minu2):
        num_match_init = num_minu1 * num_minu2

    sorted_ind = norm_simi[::-1].argsort()
    sorted_norm_simi = norm_simi[::-1].sort()
    subscripts = np.unravel_index(sorted_ind[1:num_match_init],
                                  np.shape(minu_simi))
    result['minu_index_init1'] = subscripts[:,0]
    result['minu_index_init2'] = subscripts[:,1]
    result['simi_init'] = minu_simi[sorted_ind[1:num_match_init]]

    corr = [subscripts[:,0],subscripts[:,1]]

    match_info = {}
    match_info['minu_simi'] = minu_simi
    match_info['sorted_norm_simi'] = sorted_norm_simi
    match_info['sorted_ind'] =  sorted_ind

    corr, local_compat = LSS_R_Fast2(minu_simi, corr, template1.minutiae,
                                     template2.minutiae)
    corr_t, local_compat_t = TensorMatching2(minu_simi, corr, template1.minutiae,
                                     template2.minutiae)

    result['LSS_R_minu_index1'] = corr[:,0]
    result['LSS_R_minu_index2'] = corr[:,1]
    result['LSS_R_simi'] = minu_simi[np.ravel_multi_index((corr[:,0],corr[:,1]),
                                                          [num_minu1, num_minu2])]
    result['local_compat'] = local_compat

    result['LSS_T_minu_index1'] = corr_t[:,0]
    result['LSS_T_minu_index2'] = corr_t[:,1]
    result['LSS_T_simi'] = minu_simi[np.ravel_multi_index((corr_t[:,0],corr_t[:,1]),
                                                          [num_minu1, num_minu2])]
    result['local_compat_t'] = local_compat_t

    patch_simi = ExtentPatchCorrespondences(result, template1, template2, minu_simi)

    result['patch_simi'] = patch_simi

    return result

def GetMCCSimilarity(template1,template2):
    num_minu1 = len(template1.minutiae)
    num_minu2 = len(template2.minutiae)

    minu_simi = np.zeros((num_minu1,num_minu2))

    for i in range(num_minu1):
        for j in range(num_minu2):
            m = template1.mask[i,:] & template2.mask[j,:]
            if sum(m) < len(template1.mask)*0.1:
                minu_simi[i,j] = 0
            else:
                maskN = sum(m) * 5
                m = [m,m,m,m,m]
                minu_simi[i,j] = (maskN - sum((template1.mcc[i,m] \
                                    | template2.mcc[j,m])))/(maskN + 1)
    return minu_simi

def GetDNNSimilarity(des1,des2,simi_parameter):
    num_patches = len(des1)
    if not num_patches == len(des2):
        print("Number of patches for descriptor pair does not match.")
    num_minu1 = np.shape(des1[0],1)
    num_minu2 = np.shape(des2[0],1)
    patch_type = simi_parameter['patch_type']

    if len(patch_type) == 1 and patch_type <= num_patches:
        minu_simi = np.transpose(des1[patch_type])*des2[patch_type]
    elif len(patch_type) == 1 and patch_type > num_patches:
        #use all patches
        minu_simi = np.zeros((num_minu1,num_minu2,num_patches))
        for patch_type in range(num_patches):
            minu_simi[:,:,patch_type] = np.transpose(des1[patch_type]) \
                                         * des2[patch_type]
        minu_simi = np.mean(minu_simi,2)
    return minu_simi
