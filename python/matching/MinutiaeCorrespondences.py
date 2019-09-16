import numpy as np
from scipy.spatial.distance import cdist
from NormalizeScoreMatrix import NormalizeScoreMatrix
from LSS_R_Fast2 import LSS_R_Fast2
from TensorMatching2 import TensorMatching2
from OF_SimilarityEstimation import OF_SimilarityEstimation
from sigmoid import sigmoid
from StructuralSimilarity import StructuralSimilarity
from ApplyWeights import ApplyWeights

def MinutiaeCorrespondences(template1,template2,options):

    num_minu1 = len(template1.minutiae)
    num_minu2 = len(template2.minutiae)

    result = {}
    result['num_minu1'] = num_minu1
    result['num_minu2'] = num_minu2

    minu_simi = np.zeros((num_minu1,num_minu2))

    if num_minu1 < 3 or num_minu2 < 3:
        return
    if max(template1.minutiae[:,0]) > 1000 or max(template2.minutiae[:,0]) > 1000:
        result = {}
        return

    if 'method' in options and options['method'] == 'MCC':
        minu_simi = GetMCCSimilarity(template1,template2)
    elif 'method' in options and options['method'] == 'DNN':
        simi_parameter = options['simi_parameter']
        simi_parameter['patch_types'] = options['patch_types']
        minu_simi_ridge = GetDNNSimilarity(template1.des,template2.des,
                                           simi_parameter)
        minu_simi = minu_simi_ridge
    elif 'method' in options and options['method'] == 'PCA':
        minu_simi = 3 - cdist(template1.des_PCA,template2.des_PCA)

    norm_simi = NormalizeScoreMatrix(minu_simi)

    if 'num_match_init' in options:
        num_match_init = options['num_match_init']
    else:
        num_match_init = 120

    if num_match_init > (num_minu1 * num_minu2):
        num_match_init = num_minu1 * num_minu2

    norm_simi_vector = np.reshape(norm_simi,(num_minu1 * num_minu2), order='F')
    sorted_ind = np.transpose(np.argsort(norm_simi_vector, axis=0)[::-1])
    sorted_norm_simi = norm_simi_vector[sorted_ind]
    subscripts = np.unravel_index(sorted_ind[0:num_match_init],
                                  np.shape(np.transpose(minu_simi)))
    result['minu_index_init1'] = subscripts[1]
    result['minu_index_init2'] = subscripts[0]
    init_simi_vector = np.reshape(minu_simi,(num_minu1 * num_minu2), order='F')
    result['simi_init'] = init_simi_vector[sorted_ind[0:num_match_init]]

    corr = [subscripts[1],subscripts[0]]

    match_info = {}
    match_info['minu_simi'] = minu_simi
    match_info['sorted_norm_simi'] = sorted_norm_simi
    match_info['sorted_ind'] =  sorted_ind

    corr, local_compat = LSS_R_Fast2(minu_simi, corr, template1.minutiae,
                                     template2.minutiae)
    corr_temp = corr[:]
    corr_t, local_compat_t = TensorMatching2(minu_simi, corr_temp, template1.minutiae,
                                     template2.minutiae)

    result['LSS_R_minu_index1'] = corr[0]
    result['LSS_R_minu_index2'] = corr[1]
    result['LSS_R_simi'] = init_simi_vector[np.ravel_multi_index((corr[1],corr[0]),
                                                          [num_minu2, num_minu1])]
    result['local_compat'] = local_compat

    result['LSS_T_minu_index1'] = corr_t[0]
    result['LSS_T_minu_index2'] = corr_t[1]
    result['LSS_T_simi'] = init_simi_vector[np.ravel_multi_index((corr_t[1],corr_t[0]),
                                                          [num_minu2, num_minu1])]
    result['local_compat_t'] = local_compat_t

    result = OF_SimilarityEstimation(result, template1, template2, options)

    result['minutiae1'] = template1.minutiae
    result['minutiae2'] = template2.minutiae

    Q_1 = MinutiaeQuality(template1.minutiae)
    Q_2 = MinutiaeQuality(template2.minutiae)

    minu_str_simi = StructuralSimilarity(corr, template1.minutiae,
                                         template2.minutiae, Q_1, Q_2)
    result['minu_str_simi'] = minu_str_simi

    return result

#NOT TESTED!
def GetMCCSimilarity(template1,template2):
    num_minu1 = len(template1.minutiae)
    num_minu2 = len(template2.minutiae)

    simi = np.zeros((num_minu1,num_minu2))

    for i in range(num_minu1):
        for j in range(num_minu2):
            m = template1.mccmask[i,:] & template2.mccmask[j,:]
            if sum(m) < len(template1.mccmask)*0.1:
                simi[i,j] = 0
            else:
                maskN = sum(m) * 5
                m = [m,m,m,m,m]
                simi[i,j] = (maskN - sum((template1.mcc[i,m] \
                                    | template2.mcc[j,m])))/(maskN + 1)
    return simi

def GetDNNSimilarity(des1,des2,simi_parameter):
    num_patches = len(des1)
    if not num_patches == len(des2):
        print("Number of patches for descriptor pair does not match.")
    num_minu1 = np.shape(des1[0])[0]
    num_minu2 = np.shape(des2[0])[0]
    patch_types = simi_parameter['patch_types']

    if len(patch_types) == 1 and patch_types <= num_patches:
        simi = np.transpose(des1[patch_types])*des2[patch_types]
        simi = (simi + 1) / 2
    elif len(patch_types) == 1 and patch_types > num_patches:
        #use all patches
        simi = np.zeros((num_minu1,num_minu2,num_patches))
        for patch_type in range(num_patches):
            simi[:,:,patch_type] = (np.transpose(des1[patch_type]) \
                                         * des2[patch_type] + 1) / 2
        simi = np.mean(simi,2)
    elif len(patch_types) > 1:
        simi = np.zeros((num_minu1, num_minu2, len(patch_types)))
        for i in range(len(patch_types)):
            simi[:,:,i] = (np.dot(des1[patch_types[i]-1],np.transpose(des2[patch_types[i]-1]))+1)/2
        simi = np.mean(simi, 2)
    return simi

def MinutiaeQuality(minutiae):
    D = ApplyWeights(minutiae[:,0:2], np.transpose(minutiae[:,0:2]))
    minu_num = np.shape(minutiae)[0]
    Q = np.zeros((minu_num, 1))
    for i in range(minu_num):
        d = np.sort(D[i,:])
        nd = sum(d[1:3])/2
        Q[i] = sigmoid(nd, 20, 0.05)
    return Q
