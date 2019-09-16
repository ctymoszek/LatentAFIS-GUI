from Bin2Template_Byte_TF import Bin2Template_Byte_TF
import multiprocessing
import os
from LatentMatching_OneRolled import LatentMatching_OneRolled
from functools import partial
from numpy import mean, argmin, asarray, argsort

def LatentMatching_Batch(latent_template_files,rolled_template_files,score_path,
                         num_workers,patch_types):

    #load all the latent templates
    num_latents = len(latent_template_files)
    latent_templates = []

    isLatent = 1
    for i in range(num_latents):
        temp_template = Bin2Template_Byte_TF(latent_template_files[i],isLatent)
        latent_templates.append(temp_template)
#        for j in range(len(latent_templates[i].minu_template)):
#            flag = latent_templates[i].minu_template[j].minutiae[:,3] > 0.1
#            latent_templates[i].minu_template[j].minutiae = \
#                latent_templates[i].minu_template[j].minutiae[flag,:]
#            for k in range(len(latent_templates[i].minu_template[j].des)):
#                latent_templates[i].minu_template[j].des[k] = \
#                    latent_templates[i].minu_template[j].des[k][flag,:]

    num_rolled = len(rolled_template_files)

    isLatent = 0
    rank_list = []
    corr_list = []
    if num_workers>1:
        myPool = multiprocessing.Pool(processes=num_workers)
        PoolProcess_Partial = partial(PoolProcess,
                                      rolled_template_files=rolled_template_files,
                                      score_path=score_path,
                                      latent_templates=latent_templates,
                                      patch_types=patch_types,
                                      latent_template_files=latent_template_files,
                                      isLatent=0)
        print(myPool.map(PoolProcess_Partial, range(num_rolled)))
        myPool.close()
        myPool.join()
    else:
        for i in range(num_rolled):
            rolled_template = Bin2Template_Byte_TF(rolled_template_files[i],isLatent)
            head,tail = os.path.split(rolled_template_files[i])
            root,ext = os.path.splitext(tail)
            score_file = score_path + root + '.csv'
            score, corr = LatentMatching_OneRolled(latent_templates,rolled_template,
                                          score_file,patch_types)
            if len(rank_list) < 24:
                rank_list.append([tail, score[0], score[1], mean(score)])
                corr_list.append(corr)
            elif mean(score) > asarray(rank_list)[:,3].astype(float).min():
                ind = asarray(rank_list)[:,3].astype(float).argmin()
                rank_list[ind] = [tail, score[0], score[1], mean(score)]
                corr_list[ind] = corr
        #rank_list.sort(key = lambda x: x[3], reverse=True)
        sorted_indexes = sorted(range(len(rank_list)), key = lambda x: rank_list[x][3], reverse=True)
        rank_list = [rank_list[i] for i in sorted_indexes]
        corr_list = [corr_list[i] for i in sorted_indexes[0:6]]

    return rank_list, corr_list

def PoolProcess(i,rolled_template_files,score_path,latent_templates,
                       patch_types,latent_template_files,isLatent):
    print("i: ", i)
    rolled_template = Bin2Template_Byte_TF(rolled_template_files[i],isLatent)
    head,tail = os.path.split(rolled_template_files[i])
    root,ext = os.path.splitext(tail)
    score_file = score_path + root + '.csv'
    LatentMatching_OneRolled(latent_templates,rolled_template,score_file,patch_types)

    return i
