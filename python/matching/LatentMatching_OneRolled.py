import os
import numpy as np
import csv
import scipy.io as sio
from MinutiaeCorrespondences import MinutiaeCorrespondences
from CalScores import CalScores
from PatchCorrespondences import PatchCorrespondences

def LatentMatching_OneRolled(latent_templates,rolled_template,score_file,patch_types):

    if(len(locals())<4):
        patch_types = range(1,15)

    num_latents = len(latent_templates)

    if len(rolled_template.texture_template[0].minutiae) == 0:
        score = np.zeros((num_latents,len(patch_types),2))
    else:
        score = np.zeros((num_latents,len(patch_types),3))

    options = {}
    options['method'] = 'DNN'
    options['simi_parameter'] = {}
    options['simi_parameter']['method'] = 'cos'
    options['debug'] = 0

    corr = []
    testing = range(len(patch_types))
    for k in range(len(patch_types)):
        options['patch_types'] = patch_types[k]
        for i in range(num_latents):
            rolled_minu = rolled_template.minu_template[0]
            for j in range(len(latent_templates[i].minu_template)):
                latent_minu = latent_templates[i].minu_template[j]
                result = MinutiaeCorrespondences(latent_minu,rolled_minu,options)
                corr.append(result['LSS_R_minu_index1'])
                corr.append(result['LSS_R_minu_index2'])
                method = 1
                score[i,k,j] = CalScores(result,method)

            if len(rolled_template.texture_template[0].minutiae) > 0:
                latent_tex = latent_templates[i].texture_template
                rolled_tex = rolled_template.texture_template
                result = PatchCorrespondences(latent_tex,rolled_tex,options)
                method = 3
                score[i,k,j+1] = CalScores(result,method)

    return score[:,0,:][0], corr
