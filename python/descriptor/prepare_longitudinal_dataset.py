from skimage import data, io
import numpy as np
import glob
import matplotlib.pylab as plt
from skimage.morphology import skeletonize, square, dilation
import math
from skimage.morphology import square
from skimage.transform import rescale, resize
import sys
import timeit
sys.path.append('../OF')
sys.path.append('..')
sys.path.append('../descriptor/CNN/evaluation')
sys.path.append('../enhancement')
sys.path.append('../minutiae')
sys.path.append('../utils')
sys.path.append('../minutiae/UNet/')
import get_maps
import preprocessing
import filtering, binarization
import crossnumber
import descriptor
import os
import template
import minutiae_AEC
import show

class FeatureExtraction_Rolled:
    def __init__(self,patch_types=None,des_model_dirs=None,minu_model_dir=None):
        self.des_models = None
        self.patch_types = patch_types
        self.minu_model = None
        self.minu_model_dir = minu_model_dir
        self.des_model_dirs = des_model_dirs

        self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(
            ori_num=48)

        if self.minu_model_dir is not None:
            self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))

        patchSize = 160
        oriNum = 64
        self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

        if self.des_model_dirs is not None:
            self.des_models = []
            for model_dir in self.des_model_dirs:
                self.des_models.append(descriptor.ImportGraph(model_dir))


            # self.minu_models = []
            # for model_dir in self.des_model_dirs:
            #     self.minu_models.append(descriptor.ImportGraph(model_dir))
    def feature_extraction_single_rolled(self,img_file):
        block_size = 16

        img = io.imread(img_file)
        h, w = img.shape
        mask = get_maps.get_quality_map_intensity(img)
        mnt = self.minu_model.run(img, minu_thr=0.2)

        # show.show_minutiae(img,mnt)
        des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models, self.patchIndexV,
                                                        batch_size=128)

        #texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)

        start = timeit.default_timer()
        dir_map, _ = get_maps.get_maps_STFT(img, patch_size=64, block_size=block_size, preprocess=True)
        stop = timeit.default_timer()
        print stop - start
        blkH, blkW = dir_map.shape



        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt, des=des, oimg=dir_map,
                                              mask=mask)

        rolled_template = template.Template()
        rolled_template.add_minu_template(minu_template)

        return rolled_template

    def feature_extraction(self,img_path,img_type='bmp',template_path=None):


        img_files = glob.glob(img_path+'*.'+img_type)
        assert(len(img_files)>0)

        img_files.sort()

        for i, img_file in enumerate(img_files):
            if i>20:
                break
            start = timeit.default_timer()
            rolled_template = self.feature_extraction_single_rolled(img_file)
            stop = timeit.default_timer()

            print stop - start
            if template_path is not None:
                img_name = os.path.basename(img_file)
                fname = template_path + os.path.splitext(img_name)[0] + '.dat'
                template.Template2Bin_Byte_TF(fname, rolled_template, isLatent=True)

    def feature_extraction_MSP(self,img_path, N=10000, template_path=None):

        assert(N>0)
        for i in range(1,N+1):
            #if i<=1113:
            #    continue
            start = timeit.default_timer()
            img_file = os.path.join(img_path, str(i)+'.bmp')
            rolled_template = self.feature_extraction_single_rolled(img_file)
            stop = timeit.default_timer()

            print stop - start
            if template_path is not None:
                img_name = os.path.basename(img_file)
                fname = template_path + os.path.splitext(img_name)[0] + '.dat'
                template.Template2Bin_Byte_TF(fname, rolled_template, isLatent=True)

    def feature_extraction_longitudinal_batch(self,img_path, img_type = 'bmp', N=100, texture_path=None,enhanced_path=None, template_path = None):

        subjects = glob.glob(img_path+'MI*')
        n = 0
        for subject in subjects:
            img_list = glob.glob(os.path.join(subject,'*.bmp'))
            if len(img_list)<N:
                continue
            print n
            # else:
            #     n = n + 1
            #     continue
            subjectID = os.path.basename(subject)
            texture_folder = os.path.join(texture_path,subjectID)
            if not os.path.exists(texture_folder):
                os.makedirs(texture_folder)
            enhanced_folder = os.path.join(enhanced_path, subjectID)
            if not os.path.exists(enhanced_folder):
                os.makedirs(enhanced_folder)
            template_folder = os.path.join(template_path, subjectID)
            if not os.path.exists(template_folder):
                os.makedirs(template_folder)
            for img_file in img_list:
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                template_name = os.path.join(template_folder, img_name + '.dat')
                #if os.path.exists(template_name):
                #    continue
                #fname = os.path.basename(img_file)

                rolled_template, texture_img, enh_constrast_img = self.feature_extraction_longitudinal(img_file)
                texture_img = np.asarray(texture_img,dtype=np.uint8)
                enh_constrast_img = np.asarray(enh_constrast_img, dtype=np.uint8)
                io.imsave(os.path.join(texture_folder,img_name+'.jpeg'),texture_img)
                io.imsave(os.path.join(enhanced_folder, img_name+'.jpeg'), enh_constrast_img)
                if template_path is not None:
                    #img_name = os.path.basename(img_file)
                    #fname = os.path.join(template_folder,  img_name+'.dat')
                    template.Template2Bin_Byte_TF(template_name, rolled_template, isLatent=True)
            n = n +1
        print n
        #
        # assert(N>0)
        # for i in range(1,N+1):
        #     #if i<=1113:
        #     #    continue
        #     start = timeit.default_timer()
        #     img_file = os.path.join(img_path, str(i)+'.bmp')
        #     rolled_template = self.feature_extraction_single_rolled(img_file)
        #     stop = timeit.default_timer()
        #
        #     print stop - start
        #     if template_path is not None:
        #         img_name = os.path.basename(img_file)
        #         fname = template_path + os.path.splitext(img_name)[0] + '.dat'
        #         template.Template2Bin_Byte_TF(fname, rolled_template, isLatent=True)

    def feature_extraction_longitudinal(self, img_file):
        block_size = 16

        img = io.imread(img_file)
        #print img.shape
        img = preprocessing.adjust_image_size(img,block_size)
        h, w = img.shape
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)

        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)

        mask = get_maps.get_quality_map_intensity(img)
        #show.show_mask(mask, img, fname=None, block=True)
        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(texture_img, self.dict_all, self.dict_ori,
                                                                      self.dict_spacing, block_size=16, process=False)

        enh_constrast_img = filtering.gabor_filtering_pixel(contrast_img_guassian, dir_map + math.pi / 2, fre_map,
                                                            mask=np.ones((h, w), np.int),
                                                            block_size=16, angle_inc=3)


        mnt = self.minu_model.run(img, minu_thr=0.2)

        #show.show_minutiae(img,mnt)
        des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models,
                                                        self.patchIndexV,
                                                        batch_size=128)

        blkH, blkW = dir_map.shape

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt, des=des, oimg=dir_map,
                                              mask=mask)

        rolled_template = template.Template()
        rolled_template.add_minu_template(minu_template)

        return rolled_template, texture_img, enh_constrast_img

if __name__ == '__main__':
    # args = parse_arguments(sys.argv[1:])
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    # des_model_dirs = []
    # patch_types = []
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_1/20171206-093749/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(1)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_8/20171207-160445/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(8)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_11/20171207-143926/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(11)

    #minu_model_dir = '/home/kaicao/Dropbox/Share/models/minutiae_AEC_128_fcn_aug2/'

    # for PRIP server

    des_model_dirs = []
    patch_types = []
    model_dir = '/research/prip-kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_1/20171206-093749/'
    des_model_dirs.append(model_dir)
    patch_types.append(1)

    minu_model_dir = '/research/prip-kaicao/Dropbox/Share/models/minutiae_AEC_128_fcn_aug2/'
    template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP/version_2/'

    LF_rolled = FeatureExtraction_Rolled(patch_types = patch_types, des_model_dirs = des_model_dirs, minu_model_dir = minu_model_dir)
    #LF_rolled = FeatureExtraction_Rolled()

    #LF_rolled.feature_extraction(img_path,img_type,template_path=template_path)

    #img_path = '/media/kaicao/data2/Data/MSP_background/images/'
    #img_path = '/future/Data/Rolled/NSITSD14/Image2/'
    #img_path = '/future/Data/Fingerprint/L/image/'
    img_path = '/user/pripshare/Databases/FingerprintDatabases/Longitudinal/image/'
    texture_path = '/research/prip-kaicao/AutomatedLatentRecognition/descriptors/Longitudinal/exture_image/'

    enhanced_path = '/research/prip-kaicao/AutomatedLatentRecognition/descriptors/Longitudinal/enhanced_image/'
    template_path = '/research/prip-kaicao/AutomatedLatentRecognition/descriptors/Longitudinal/template/'
    img_type = 'bmp'
    LF_rolled.feature_extraction_longitudinal_batch(img_path,img_type=img_type,texture_path=texture_path,enhanced_path=enhanced_path,template_path=template_path)
    #LF_rolled.feature_extraction_MSP(img_path,N=1114,template_path=template_path)
