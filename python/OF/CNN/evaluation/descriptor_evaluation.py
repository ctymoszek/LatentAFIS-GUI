import glob
import cv2
import descriptor
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import tensorflow as tf
import argparse
import os
import sys
sys.path.append('../src/')
import facenet
from sklearn import metrics
import scipy.io as sio


def show_features(img,minutiae,ROI=None,fname=None):
    # for the latent or the low quality rolled print
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    R = 10
    arrow_len = 15
    if ROI is not None:
        h,w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i,j] == 0:
                    img[i,j] = 255

    ax.imshow(img, cmap='gray')
    minu_num = len(minutiae)
    for i in range(0, minu_num):
        xx = minutiae[i][0]
        yy = minutiae[i][1]
        circ = Circle((xx, yy), R, color='r', fill=False)
        ax.add_patch(circ)

        ori = -minutiae[i][2]
        dx = math.cos(ori) * arrow_len
        dy = math.sin(ori) * arrow_len
        ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
    plt.show()
    if fname is not None:
        fig.savefig(fname,dpi = 600)

def feature_patches(img_files, minu_files, output_path):


    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

    n = 0
    for minu_file, img_file in zip(minu_files,img_files):
        img = cv2.imread(img_file)
        #cv2.imshow("latent", latent_img)
        #cv2.imshow("rolled", rolled_img)

        minutiae = np.loadtxt(minu_file, dtype='f', delimiter=',')
        minutiae[:,2] =  minutiae[:,2] - math.pi/2
        #show_features(latent_img, latent_minutiae, ROI=None, fname=None)
        minutiae[:, 2] = -minutiae[:, 2]

        patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=6)


        for j in range(len(patches)):

            fname = "%05d" % n + '.jpeg'
            n = n + 1
            cv2.imwrite(output_path+fname, patches[j])
        #cv2.imshow("rolled patch", rolled_patches[0]/255)

        #cv2.waitKey(0)
    #np.save(output_file, features)

def feature_extraction(img_files, minu_files, model_dir,output_file,patch_type = 6):


    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)


    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Get the paths for the corresponding images
            # paths, actual_issame = lfw.get_paths(os.path.expanduser(args.test_dir), pairs, args.lfw_file_ext)
            # Load the model
            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("batch_join:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # images_placeholder = tf.get_default_graph4().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("Add:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            features = np.array([], dtype=np.float32).reshape(0, embedding_size)
            for minu_file, img_file in zip(minu_files,img_files):
                img = cv2.imread(img_file)
                #cv2.imshow("latent", latent_img)
                #cv2.imshow("rolled", rolled_img)

                minutiae = np.loadtxt(minu_file, dtype='f', delimiter=',')
                minutiae[:,2] =  minutiae[:,2] - math.pi/2
                #show_features(latent_img, latent_minutiae, ROI=None, fname=None)
                #minutiae[:, 2] = -minutiae[:, 2]

                patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=patch_type)
                # for i in range(len(patches)):
                #     patch = patches[i, :, :, 0]
                #     plt.imshow(patch, cmap='gray')
                #     plt.show()

                feed_dict = {images_placeholder: patches, phase_train_placeholder: False}
                latent_emb = sess.run(embeddings, feed_dict=feed_dict)


                features = np.vstack([features, latent_emb])

                #cv2.imshow("latent patch", latent_patches[0]/255)
                #cv2.imshow("rolled patch", rolled_patches[0]/255)

                #cv2.waitKey(0)
            np.save(output_file, features)


def feature_evaluation(latent_features, rolled_features):

    assert (latent_features.shape[0] == rolled_features.shape[0])

    minu_num = latent_features.shape[0]
    # feature normalization
    for i in range(minu_num):
        norm = np.linalg.norm(latent_features[i, :]) + 0.0000001
        latent_features[i, :] = latent_features[i, :] / norm

        norm = np.linalg.norm(rolled_features[i, :]) + 0.0000001
        rolled_features[i, :] = rolled_features[i, :] / norm

    score = (np.matmul(latent_features, rolled_features.transpose()) + 1)/2
    # genuine score computation
    gen_score = score.diagonal()

    ind_upper = np.triu_indices(minu_num, 1)
    ind_1ower = np.tril_indices(minu_num, -1)

    # impostor score computation
    imp_score = np.hstack((score[ind_upper],score[ind_1ower]))

    imp_num = len(imp_score)
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((np.ones((minu_num,)), np.zeros((imp_num,))))
    fpr, tpr, thrsholds = metrics.roc_curve(y, score, pos_label=1)
    return fpr, tpr
    #plt.plot(fpr,tpr)
    #plt.xscale('log')
    #plt.xlim([10**(-4),1])
    #plt.show()

def feature_evaluation_fusion(latent_features, rolled_features):
    assert (len(latent_features) == len(rolled_features))
    assert (latent_features[0].shape[0] == rolled_features[0].shape[0])

    minu_num = latent_features[0].shape[0]
    des_num = len(latent_features)
    # feature normalization
    for k in range(des_num):
        for i in range(minu_num):
            norm = np.linalg.norm(latent_features[k][i, :]) + 0.0000001
            latent_features[k][i, :] = latent_features[k][i, :] / norm

            norm = np.linalg.norm(rolled_features[k][i, :]) + 0.0000001
            rolled_features[k][i, :] = rolled_features[k][i, :] / norm
    score = 0
    for k in range(des_num):
        score += (np.matmul(latent_features[k], rolled_features[k].transpose()) + 1.0 )/2.0
    # genuine score computation
    gen_score = score.diagonal()

    ind_upper = np.triu_indices(minu_num, 1)
    ind_1ower = np.tril_indices(minu_num, -1)

    # impostor score computation
    imp_score = np.hstack((score[ind_upper],score[ind_1ower]))

    imp_num = len(imp_score)
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((np.ones((minu_num,)), np.zeros((imp_num,))))
    fpr, tpr, thrsholds = metrics.roc_curve(y, score, pos_label=1)
    return fpr, tpr

def evaluation_2():



    rolled_features_new = []
    latent_features_new = []

    patch_type = 6
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    rolled_features_new.append(np.load('Data/original_rolled_'+ext+'.npy'))
    latent_features_new.append(np.load('Data/original_latent_'+ext+'.npy'))

    patch_type = 6
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    rolled_features_new.append(np.load('Data/original_rolled_' + ext + '.npy'))
    latent_features_new.append(np.load('Data/enhanced_latent_' + ext + '.npy'))



    rolled_features_PAMI = []
    latent_features_PAMI = []
    rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_6.mat')['features'])
    latent_features_PAMI.append(sio.loadmat('Data/PAMI/original_latent_features_patch_type_6.mat')['features'])

    rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_6.mat')['features'])
    latent_features_PAMI.append(sio.loadmat('Data/PAMI/enhanced_latent_features_patch_type_6.mat')['features'])



    for i in range(len(latent_features_new)):
        fpr_1, tpr_1 = feature_evaluation(latent_features_new[i], rolled_features_new[i])
        ind = np.where(fpr_1 < 0.001)
        print tpr_1[ind[0][-1]]

        fpr_2, tpr_2 = feature_evaluation(latent_features_PAMI[i], rolled_features_PAMI[i])
        ind = np.where(fpr_2 < 0.001)
        print tpr_2[ind[0][-1]]


    fpr_1, tpr_1 = feature_evaluation_fusion(latent_features_new, rolled_features_new)
    ind = np.where(fpr_1 < 0.001)
    print tpr_1[ind[0][-1]]
    fpr_2, tpr_2 = feature_evaluation_fusion(latent_features_PAMI, rolled_features_PAMI)
    ind = np.where(fpr_2 < 0.001)
    print tpr_2[ind[0][-1]]

    legend = ['New descriptor', 'PAMI descriptor']
    #legend = ['original latent Inception','enhanced latent Inception','original latent AlexNet','enhanced latent AlexNet']
    plt.plot(fpr_1, tpr_1)
    plt.plot(fpr_2, tpr_2)
    # plt.plot(fpr_3, tpr_3)
    # plt.plot(fpr_4, tpr_4)
    plt.xscale('log')
    plt.xlim([10**(-4),1])

    plt.legend(legend,loc='upper right')
    plt.show()

def evaluation_1():

    patch_type = 11
    ext = 'normalized_raw_enhanced_patchtype_' +str(patch_type)
    rolled_features_new = []
    latent_features_new = []
    rolled_features_new.append(np.load('Data/original_rolled_'+ext+'.npy'))
    latent_features_new.append(np.load('Data/enhanced_latent_'+ext+'.npy'))

    patch_type = 6
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    rolled_features_new.append(np.load('Data/original_rolled_' + ext + '.npy'))
    latent_features_new.append(np.load('Data/enhanced_latent_' + ext + '.npy'))

    patch_type = 8
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    rolled_features_new.append(np.load('Data/original_rolled_' + ext + '.npy'))
    latent_features_new.append(np.load('Data/enhanced_latent_' + ext + '.npy'))


    # patch_type = 11
    # ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    # rolled_features_new.append(np.load('Data/original_rolled_' + ext + '.npy'))
    # latent_features_new.append(np.load('Data/enhanced_latent_' + ext + '.npy'))



    rolled_features_PAMI = []
    latent_features_PAMI = []
    rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_11.mat')['features'])
    latent_features_PAMI.append(sio.loadmat('Data/PAMI/enhanced_latent_features_patch_type_11.mat')['features'])

    rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_6.mat')['features'])
    latent_features_PAMI.append(sio.loadmat('Data/PAMI/enhanced_latent_features_patch_type_6.mat')['features'])

    rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_8.mat')['features'])
    latent_features_PAMI.append(sio.loadmat('Data/PAMI/enhanced_latent_features_patch_type_8.mat')['features'])

    # rolled_features_PAMI.append(sio.loadmat('Data/PAMI/original_rolled_features_patch_type_11.mat')['features'])
    # latent_features_PAMI.append(sio.loadmat('Data/PAMI/enhanced_latent_features_patch_type_11.mat')['features'])



    for i in range(len(latent_features_new)):
        fpr_1, tpr_1 = feature_evaluation(latent_features_new[i], rolled_features_new[i])
        ind = np.where(fpr_1 < 0.001)
        print tpr_1[ind[0][-1]]

        fpr_2, tpr_2 = feature_evaluation(latent_features_PAMI[i], rolled_features_PAMI[i])
        ind = np.where(fpr_2 < 0.001)
        print tpr_2[ind[0][-1]]


    fpr_1, tpr_1 = feature_evaluation_fusion(latent_features_new, rolled_features_new)
    ind = np.where(fpr_1 < 0.001)
    print tpr_1[ind[0][-1]]
    fpr_2, tpr_2 = feature_evaluation_fusion(latent_features_PAMI, rolled_features_PAMI)
    ind = np.where(fpr_2 < 0.001)
    print tpr_2[ind[0][-1]]

    legend = ['New descriptor', 'PAMI descriptor']
    #legend = ['original latent Inception','enhanced latent Inception','original latent AlexNet','enhanced latent AlexNet']
    plt.plot(fpr_1, tpr_1)
    plt.plot(fpr_2, tpr_2)
    # plt.plot(fpr_3, tpr_3)
    # plt.plot(fpr_4, tpr_4)
    plt.xscale('log')
    plt.xlim([10**(-4),1])

    plt.legend(legend,loc='upper right')
    plt.show()


def compare_rotation():

    patch_type = 8
    ext = 'normalized_raw_enhanced_rotate_patchtype_' +str(patch_type)
    rolled_features_new = []
    latent_features_new = []
    rolled_features_new.append(np.load('Data/original_rolled_'+ext+'.npy'))
    latent_features_new.append(np.load('Data/enhanced_latent_'+ext+'.npy'))

    rolled_features_PAMI = []
    latent_features_PAMI = []
    patch_type = 8
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    rolled_features_PAMI.append(np.load('Data/original_rolled_' + ext + '.npy'))
    latent_features_PAMI.append(np.load('Data/enhanced_latent_' + ext + '.npy'))




    for i in range(len(latent_features_new)):
        fpr_1, tpr_1 = feature_evaluation(latent_features_new[i], rolled_features_new[i])
        ind = np.where(fpr_1 < 0.001)
        print tpr_1[ind[0][-1]]

        fpr_2, tpr_2 = feature_evaluation(latent_features_PAMI[i], rolled_features_PAMI[i])
        ind = np.where(fpr_2 < 0.001)
        print tpr_2[ind[0][-1]]


    # fpr_1, tpr_1 = feature_evaluation_fusion(latent_features_new, rolled_features_new)
    # ind = np.where(fpr_1 < 0.001)
    # print tpr_1[ind[0][-1]]
    # fpr_2, tpr_2 = feature_evaluation_fusion(latent_features_PAMI, rolled_features_PAMI)
    # ind = np.where(fpr_2 < 0.001)
    # print tpr_2[ind[0][-1]]

    legend = ['New descriptor', 'PAMI descriptor']
    #legend = ['original latent Inception','enhanced latent Inception','original latent AlexNet','enhanced latent AlexNet']
    plt.plot(fpr_1, tpr_1)
    plt.plot(fpr_2, tpr_2)
    # plt.plot(fpr_3, tpr_3)
    # plt.plot(fpr_4, tpr_4)
    plt.xscale('log')
    plt.xlim([10**(-4),1])

    plt.legend(legend,loc='upper right')
    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--template_path', type=str,
                        help='template directory',
                        default='/media/kaicao/Data/AutomatedLatentRecognition/templates/latents/NSITSD27/Latents_3/')
    parser.add_argument('--new_template_path', type=str,
                        help='new template directory',
                        default='/media/kaicao/Data/AutomatedLatentRecognition/templates/latents/NSITSD27/Latents_3_replaced_6/')
    parser.add_argument('--image_type', type=str,
                        help='fingerprint image type, e.g., latent and rolled',
                        default='latent')
    parser.add_argument('--img_path', type=str,
        help='Path to the data directory containing aligned fingerprint images.',
                        default='/home/kaicao/Dropbox/Research/LatentMatching/CodeForPaper/Evaluation/Code/time/latent/')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/home/kaicao/Research/AutomatedLatentRecognition/models/facenet/20171127-000844/')#''../models/triplet_debug/20170709-000103/')
    parser.add_argument('--fname', type=str,
        help='file name to save feature vectors extracted by learned CNN model',
                        default='~/Dropbox/Research/Indexing/scr/facenet/feature/NISTSD4.npy')#''../models/triplet_debug/20170709-000103/')
    return parser.parse_args(argv)

if __name__=='__main__':
    # model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet/20171129-153819/'
    #model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced/20171130-142625/'

    model_dir='/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_1/20171206-093749/'

    patch_type = 11
    model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_11/20171207-143926/'

    patch_type = 8
    model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_8/20171207-160445/'
    ext = 'normalized_raw_enhanced_patchtype_' + str(patch_type)
    corr_path = '/home/kaicao/PRIP/Data/Latent/DB/NIST27/MatchedMinutiae/'


    compute_feature = True
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # extract descriptor on original latent images

    if compute_feature:
        matched_latent_minu_files = glob.glob(corr_path + 'l*.txt')
        matched_latent_minu_files.sort()
        latent_img_path = '/home/kaicao/PRIP/Data/Latent/DB/NIST27/image/'
        latent_img_files = glob.glob(latent_img_path + '*.bmp')
        latent_img_files.sort()
        feature_extraction(latent_img_files, matched_latent_minu_files, model_dir, 'Data/original_latent_'+ ext+'.npy',
                           patch_type=patch_type)

    # extract descriptor on enhanced latent images
    if compute_feature:
        matched_latent_minu_files = glob.glob(corr_path + 'l*.txt')
        matched_latent_minu_files.sort()
        latent_img_path = '/home/kaicao/Dropbox/Research/LatentMatching/CodeForPaper/Evaluation/Code/time/latent/'
        latent_img_files = glob.glob(latent_img_path + '*.bmp')
        latent_img_files.sort()
        feature_extraction(latent_img_files, matched_latent_minu_files, model_dir, 'Data/enhanced_latent_'+ext+'.npy', patch_type=patch_type)

    # extract descriptor on original rolled images
    if compute_feature:
        matched_rolled_minu_files = glob.glob(corr_path + 'r*.txt')
        matched_rolled_minu_files.sort()
        rolled_img_path = '/home/kaicao/PRIP/Data/Rolled/NIST27/Image/'
        rolled_img_files = glob.glob(rolled_img_path + '*.bmp')
        rolled_img_files.sort()
        feature_extraction(rolled_img_files, matched_rolled_minu_files, model_dir, 'Data/original_rolled_'+ext+'.npy',
                           patch_type=patch_type)

    # extract descriptor on original rolled images
    if compute_feature:
        matched_rolled_minu_files = glob.glob(corr_path + 'r*.txt')
        matched_rolled_minu_files.sort()
        rolled_img_path = '/home/kaicao/PRIP/Data/Rolled/NIST27/Image/'
        rolled_img_files = glob.glob(rolled_img_path + '*.bmp')
        rolled_img_files.sort()
        feature_extraction(rolled_img_files, matched_rolled_minu_files, model_dir,
                           'Data/original_rolled_' + ext + '.npy',
                           patch_type=patch_type)

    compare_rotation()
    #evaluation_1()
    #feature_extraction(parse_arguments(sys.argv[1:]))
    #latent_features = np.load('Data/original_latent_'+ext+'.npy')

    # # feature_extraction(parse_arguments(sys.argv[1:]))
    # latent_features = np.load('Data/enhanced_latent_normalized_raw_enhanced.npy')
    # rolled_features = np.load('Data/original_rolled_normalized_raw_enhanced.npy')
    # fpr_1, tpr_1 = feature_evaluation(latent_features, rolled_features)
    #
    # latent_features = np.load('Data/enhanced_latent_normalized.npy')
    # rolled_features = np.load('Data/original_rolled_normalized.npy')
    # fpr_2, tpr_2 = feature_evaluation(latent_features, rolled_features)
    #
    # legend = ['enhanced latent (raw and enhanced)','enhanced latent Inception']
    # plt.plot(fpr_1, tpr_1)
    # plt.plot(fpr_2, tpr_2)
    # plt.xscale('log')
    # plt.xlim([10 ** (-4), 1])
    #
    # plt.legend(legend, loc='upper right')
    # plt.show()