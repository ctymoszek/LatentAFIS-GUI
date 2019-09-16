import glob
import platform
import os
from sys import argv
from LatentMatching_Batch import LatentMatching_Batch
import json

def RunMatcher(latent_data_fname):
    sysType = platform.system()
    dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(dir, 'Data')

    score_dir = data_path + 'scores'
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    score_path = os.path.join(score_dir, latent_data_fname.replace('.dat', '_results.txt'))
    #print(score_path)

    latent_data_fname = os.path.join(dir, 'Data/Latent', latent_data_fname)
    rolled_template_path = os.path.join(dir, 'Data/Rolled')
    rolled_template_files = glob.glob(rolled_template_path + '*.dat')
    rolled_template_files.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename))))

    num_workers = 1

    all_patches = list(range(1,4))
    selected_patches = list(range(1,4))

    patch_types = [selected_patches]

    rank_list, corr_list = LatentMatching_Batch([latent_data_fname],rolled_template_files,
                             score_dir,num_workers,patch_types)
    # print(rank_list)
    # print("rank list length: " + str(len(rank_list)))
    # print(corr_list)
    # print("corr list length: " + str(len(corr_list)))
    result_dict = {}
    for i in range(len(rank_list)):
        one_dict = {}
        one_dict['name'] = rank_list[i][0]
        one_dict['score1'] = rank_list[i][1]
        one_dict['score2'] = rank_list[i][2]
        one_dict['score_mean'] = rank_list[i][3]
        if i < 6:
            corr_dict = {}
            corr_l = {}
            corr_r = {}
            for j in range(len(corr_list[i][0])):
                corr_dict[str(corr_list[i][0][j])] = str(corr_list[i][1][j])
            one_dict['corr1'] = corr_dict
            corr_dict = {}
            corr_l = {}
            corr_r = {}
            for j in range(len(corr_list[i][2])):
                corr_dict[str(corr_list[i][2][j])] = str(corr_list[i][3][j])
            one_dict['corr2'] = corr_dict
        result_dict[str(i)] = one_dict
    #print(result_dict)
    #print(score_path)
    with open(score_path, 'w') as of:
        json.dump(result_dict, of)
        #print('dumped json')
    #print(score_path)
    return rank_list, corr_list

if __name__ == '__main__':
    latent_data_fname = argv[1]
    RunMatcher(latent_data_fname)
