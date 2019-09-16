from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import Template
import argparse
import os
import sys
sys.path.append('../src/')
import facenet
import descriptor
import sys
import math
import glob
import cv2
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def show_minutiae(img,minutiae,ROI=None,fname=None,block = True):
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
    plt.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600)
        plt.close()

def show_minutiae_sets(img,minutiae_sets,ROI=None,fname=None,block = True):
    # for the latent or the low quality rolled print
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    arrow_len = 15
    if ROI is not None:
        h,w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i,j] == 0:
                    img[i,j] = 255

    ax.imshow(img, cmap='gray')
    color = ['r','b']
    R = [10,8,6]
    for k in range(len(minutiae_sets)):
        minutiae = minutiae_sets[k]
        #minutiae = np.asarray(minutiae)
        minu_num = len(minutiae)
        for i in range(0, minu_num):
            xx = minutiae[i,0]
            yy = minutiae[i,1]
            circ = Circle((xx, yy), R[k], color=color[k], fill=False)
            ax.add_patch(circ)

            ori = -minutiae[i,2]
            dx = math.cos(ori) * arrow_len
            dy = math.sin(ori) * arrow_len
            ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc=color[k], ec=color[k])

    plt.show(block=block)
    if fname is not None:
        fig.savefig(fname,dpi = 600)
        plt.close()




def main_single(args):
    template_path = args.template_path
    img_path = args.img_path
    new_template_path = args.new_template_path
    # for latents
    if args.image_type == 'latent':
        template_files = os.listdir(template_path)
        template_files.sort()
        isLatent = 1
    assert (len(template_files) > 0)

    batch_size = args.batch_size
    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)



    #for template in template.minu_template:
    #    a = 1

    batch_size = args.batch_size
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Get the paths for the corresponding images
            # paths, actual_issame = lfw.get_paths(os.path.expanduser(args.test_dir), pairs, args.lfw_file_ext)
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("batch_join:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # images_placeholder = tf.get_default_graph4().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("Add:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on testing images')

            for file in template_files:
                template = Template.Bin2Template_Byte(template_path + file, isLatent=isLatent)
                # minutiae templates
                img_file = img_path + file.split('.')[0] + '.bmp'
                img = cv2.imread(img_file) # cv2.IMREAD_GRAYSCALE
                img = img.astype(float)
                for t in template.minu_template:
                    minutiae = t.minutiae
                    patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=6)
                    nrof_patches = len(patches)
                    emb_array = np.zeros((nrof_patches, embedding_size))
                    nrof_batches = int(math.ceil(1.0 * nrof_patches / batch_size))
                    for i in range(nrof_batches):
                        print(i)
                        start_index = i * batch_size
                        end_index = min((i + 1) * batch_size, nrof_patches)
                        patches_batch = patches[start_index:end_index,:,:]
                        feed_dict = {images_placeholder: patches_batch, phase_train_placeholder: False}
                        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                    for i in range(nrof_patches):
                        norm = np.linalg.norm(emb_array[i, :]) + 0.0000001
                        emb_array[i, :] = emb_array[i, :] / norm
                        print(i)
                        #np.save(args.fname, emb_array)


def main(args,patch_types=None,model_dirs=None,rolled_range=None):
    template_path = args.template_path
    img_path = args.img_path
    new_template_path = args.new_template_path

    if not os.path.exists(new_template_path):
        os.makedirs(new_template_path)
    # for latents
    isLatent = (args.image_type == 'latent')
    if isLatent:
        template_files = os.listdir(template_path)
        template_files.sort()
    else:
        template_files = [str(i+1)+'.dat' for i in range(rolled_range[0],rolled_range[1])]
    assert (len(template_files) > 0)

    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

    assert(len(patch_types) == len(model_dirs) )


    models = []
    for model_dir in model_dirs:
        models.append(ImportGraph(model_dir))

    #for template in template.minu_template:
    batch_size = args.batch_size
    for print_ind, file in enumerate(template_files):
        print(print_ind)
        template = Template.Bin2Template_Byte(template_path + file, isLatent=isLatent)
        if template is None:
            continue
        # minutiae templates
        img_file = img_path + file.split('.')[0] + '.bmp'
        img = cv2.imread(img_file) # cv2.IMREAD_GRAYSCALE
        img = img.astype(float)
        for n, t in enumerate(template.minu_template):
            minutiae = t.minutiae
            template.minu_template[n].des = []
            for k,patch_type in enumerate(patch_types):
                embedding_size =models[k].embedding_size
                patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=patch_type)
                # for i in range(len(patches)):
                #     patch = patches[i, :, :, 0]
                #     plt.imshow(patch, cmap='gray')
                #     plt.show()
                nrof_patches = len(patches)
                emb_array = np.zeros((nrof_patches, embedding_size))
                nrof_batches = int(math.ceil(1.0 * nrof_patches / batch_size))
                for i in range(nrof_batches):
                    #print(i)
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_patches)
                    patches_batch = patches[start_index:end_index,:,:]
                    emb_array[start_index:end_index, :] = models[k].run(patches_batch)
                for i in range(nrof_patches):
                    norm = np.linalg.norm(emb_array[i, :]) + 0.0000001
                    emb_array[i, :] = emb_array[i, :] / norm
                template.minu_template[n].des.append(emb_array)
        for n, t in enumerate(template.texture_template):
            template.texture_template[n].minutiae =[]
            #minutiae = t.minutiae
            minutiae = None

            template.texture_template[n].des = []
            continue

            for k, patch_type in enumerate(patch_types):
                embedding_size = models[k].embedding_size
                patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=patch_type)
                nrof_patches = len(patches)
                emb_array = np.zeros((nrof_patches, embedding_size))
                nrof_batches = int(math.ceil(1.0 * nrof_patches / batch_size))
                for i in range(nrof_batches):
                    #print(i)
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_patches)
                    patches_batch = patches[start_index:end_index, :, :]
                    emb_array[start_index:end_index, :] = models[k].run(patches_batch)
                for i in range(nrof_patches):
                    norm = np.linalg.norm(emb_array[i, :]) + 0.0000001
                    emb_array[i, :] = emb_array[i, :] / norm
                #template.texture_template[n].des[patch_type] = emb_array
                template.texture_template[n].des.append(emb_array)
        fname = new_template_path + file
        Template.Template2Bin_Byte_TF(fname, template, isLatent=isLatent)
                #np.save(args.fname, emb_array)

def main_new_minutiae(args, patch_types=None,model_dirs=None,rolled_range=None):
    minutiae_path = args.minutiae_path
    img_path = args.img_path
    new_template_path = args.new_template_path
    mask_path = args.mask_path
    template_path = args.template_path

    if not os.path.exists(new_template_path):
        os.makedirs(new_template_path)
    # for latents
    isLatent = (args.image_type == 'latent')
    if isLatent:
        minutiae_files = []
        for i in range(len(minutiae_path)):
            minutiae_files.append(glob.glob(minutiae_path[i]+'*.txt'))
            minutiae_files[-1].sort()
        img_files = glob.glob(img_path+'*.bmp')
        img_files.sort()
        mask_files = glob.glob(mask_path + '*.bmp')
        mask_files.sort()

        template_files = glob.glob(template_path+'*.dat')
        template_files.sort()

    else:
        template_files = [str(i+1)+'.dat' for i in range(rolled_range[0],rolled_range[1])]
    assert (len(minutiae_files) > 0)

    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

    assert(len(patch_types) == len(model_dirs) )


    models = []
    for model_dir in model_dirs:
        models.append(ImportGraph(model_dir))

    #for template in template.minu_template:
    batch_size = args.batch_size
    nrof_imgs = len(img_files)
    nrof_minutiae_set = len(minutiae_files)
    for print_ind in range(nrof_imgs):
        # minutiae templates
        img_file = img_files[print_ind]
        img = cv2.imread(img_file) # cv2.IMREAD_GRAYSCALE
        img = img.astype(float)
        #mask = cv2.imread(mask_files[print_ind], cv2.IMREAD_GRAYSCALE)

        template_file = template_files[print_ind]
        template = Template.Bin2Template_Byte(template_file, isLatent=isLatent)

        for n in range(nrof_minutiae_set):
            minutiae = np.loadtxt(minutiae_files[n][print_ind])
            nrof_minutiae = len(minutiae)
            mask = template.minu_template[0].mask
            #show_minutiae_sets(img,[minutiae],ROI=None)
            #plt.imshow(patch, cmap='gray')
            #plt.show()
            #remove minutiae in the background
            h, w = mask.shape
            flag = np.ones((nrof_minutiae,),dtype=bool)
            for i in range(nrof_minutiae):
                x =int(minutiae[i,0])
                y = int(minutiae[i,1])
                if y<10 or x<10 or x>w-10 or y>h-10:
                    flag[i] = False
                elif np.sum(mask[y-1:y+2,x-1:x+2]) == 0:
                    flag[i] = False
            minutiae = minutiae[flag,:]
            if len(minutiae)<3:
                print(len(minutiae))
            #show_minutiae_sets(img,[minutiae], ROI=None, fname=None, block=True)
            template.minu_template[n].des = []
            template.minu_template[n].minutiae = minutiae
            for k,patch_type in enumerate(patch_types):
                embedding_size =models[k].embedding_size
                patches = descriptor.extract_patches(minutiae, img, patchIndexV, patch_type=patch_type)
                # for i in range(len(patches)):
                #     patch = patches[i, :, :, 0]
                #     plt.imshow(patch, cmap='gray')
                #     plt.show()
                nrof_patches = len(patches)
                emb_array = np.zeros((nrof_patches, embedding_size))
                nrof_batches = int(math.ceil(1.0 * nrof_patches / batch_size))
                for i in range(nrof_batches):
                    #print(i)
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_patches)
                    patches_batch = patches[start_index:end_index,:,:]
                    emb_array[start_index:end_index, :] = models[k].run(patches_batch)
                for i in range(nrof_patches):
                    norm = np.linalg.norm(emb_array[i, :]) + 0.0000001
                    emb_array[i, :] = emb_array[i, :] / norm
                template.minu_template[n].des.append(emb_array)
        for n, t in enumerate(template.texture_template):
            template.texture_template[n].minutiae =[]
            #minutiae = t.minutiae
            minutiae = None

            template.texture_template[n].des = []
            continue

        fname = new_template_path + os.path.basename(template_file)
        Template.Template2Bin_Byte_TF(fname, template, isLatent=isLatent)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.',default='0')
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
        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/home/kaicao/Research/AutomatedLatentRecognition/models/facenet/20171127-000844/')#''../models/triplet_debug/20170709-000103/')
    parser.add_argument('--fname', type=str,
        help='file name to save feature vectors extracted by learned CNN model',
                        default='~/Dropbox/Research/Indexing/scr/facenet/feature/NISTSD4.npy')#''../models/triplet_debug/20170709-000103/')
    return parser.parse_args(argv)

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_dirs = []
    patch_types = []
    model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_1/20171206-093749/'
    model_dirs.append(model_dir)
    patch_types.append(1)
    model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_8/20171207-160445/'
    model_dirs.append(model_dir)
    patch_types.append(8)
    model_dir = '/home/kaicao/Research/AutomatedLatentRecognition/models/facenet_raw_enhanced_type_11/20171207-143926/'
    model_dirs.append(model_dir)
    patch_types.append(11)

    # rolled

    # args.template_path = '/home/kaicao/PRIP/LatentMatching/CodeForPaper/Evaluation/Data/Templates_patchType_1_to_14_Byte/Rolled/'
    # args.image_type = 'rolled'
    # args.img_path = '/home/kaicao/PRIP_Share/Databases/FingerprintDatabases/images/'
    # args.new_template_path = '/media/kaicao/Data/AutomatedLatentRecognition/templates_modified_Des/rolled_2_minutiae/'
    # rolled_range = [0, 10000]
    # main(args, patch_types, model_dirs, rolled_range)

    # latent
    # args.template_path = '/home/kaicao/PRIP/LatentMatching/CodeForPaper/Evaluation/Data/Templates_patchType_1_to_14_Byte/Latents_3/'
    # args.image_type = 'latent'
    # args.img_path = '/home/kaicao/Dropbox/Research/LatentMatching/CodeForPaper/Evaluation/Code/time/latent/'
    # args.new_template_path = '/media/kaicao/Data/AutomatedLatentRecognition/templates_modified_Des/latents/NSITSD27_2_minutiae/'
    # main(args,patch_types,model_dirs)

    # new latent minutiae extraction
    args.template_path = '/home/kaicao/PRIP/LatentMatching/CodeForPaper/Evaluation/Data/Templates_patchType_1_to_14_Byte/Latents_3/'
    args.minutiae_path = []
    args.minutiae_path.append('/home/kaicao/PRIP/AutomatedLatentRecognition/pred_minutiae_cylinder_aug_texture/')
    args.minutiae_path.append('/home/kaicao/PRIP/AutomatedLatentRecognition/pred_minutiae_cylinder_aug/')
    args.image_type = 'latent'
    args.img_path = '/home/kaicao/Dropbox/Research/LatentMatching/CodeForPaper/Evaluation/Code/time/latent/'
    args.new_template_path = '/media/kaicao/Data/AutomatedLatentRecognition/templates_modified_Des/latents/NSITSD27_2_new_minutiae/'
    args.mask_path = '/home/kaicao/PRIP/Data/Latent/DB/ManualInformation/NIST27/maskNIST27/'
    main_new_minutiae(args,patch_types,model_dirs)




