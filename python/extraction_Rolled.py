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
import argparse
import scipy
sys.path.append('OF')
sys.path.append('descriptor/CNN/evaluation')
sys.path.append('enhancement')
sys.path.append('minutiae')
sys.path.append('utils')
sys.path.append('minutiae/UNet/')
import get_maps
import preprocessing
import filtering
import binarization
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

        if self.minu_model_dir is not None:
            self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))

        self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(
            ori_num=24)
        patchSize = 160
        oriNum = 64
        self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

        if self.des_model_dirs is not None:
            self.des_models = []
            for model_dir in self.des_model_dirs:
                self.des_models.append(descriptor.ImportGraph(model_dir))
            self.patch_size = 96


            # self.minu_models = []
            # for model_dir in self.des_model_dirs:
            #     self.minu_models.append(descriptor.ImportGraph(model_dir))
        self.gabor_filters = filtering.get_gabor_filters()

    def feature_extraction_single_rolled_enhancement(self, img_file):
        block_size = 16

        img = io.imread(img_file)
        # print img.shape

        img = preprocessing.adjust_image_size(img, block_size)


        h, w = img.shape
        #texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)

        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)

        mask = get_maps.get_quality_map_intensity(img)



        #show.show_mask(mask, img, fname=None, block=True)
        start = timeit.default_timer()
        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(contrast_img_guassian, self.dict_all, self.dict_ori,
                                                                      self.dict_spacing, block_size=16, process=False)
        stop = timeit.default_timer()
        OF_time = stop-start
        print 'of estimate time: %f' % (OF_time)
        start = timeit.default_timer()
        enh_constrast_img = filtering.gabor_filtering_pixel2(contrast_img_guassian, dir_map + math.pi / 2, fre_map,mask=mask, gabor_filters=self.gabor_filters)
        stop = timeit.default_timer()
        filtering_time = stop - start
        print 'filtering time: %f' % (filtering_time)
        mnt = self.minu_model.run_whole_image(img, minu_thr=0.2)

        # show.show_minutiae(img,mnt)
        des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models,
                                                        self.patchIndexV,
                                                        batch_size=128)

        blkH, blkW = dir_map.shape

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt, des=des, oimg=dir_map,
                                              mask=mask)

        rolled_template = template.Template()
        rolled_template.add_minu_template(minu_template)

        # texture templates
        stride = 32
        x = np.arange(24, w - 24, stride)
        y = np.arange(24, h - 24, stride)

        virtual_minutiae = []
        distFromBg = scipy.ndimage.morphology.distance_transform_edt(mask)
        for y_i in y:
            for x_i in x:
                if (distFromBg[y_i][x_i] <= 16):
                    continue
                ofY = int(y_i / 16)
                ofX = int(x_i / 16)

                ori = -dir_map[ofY][ofX]
                # print("ori = " + str(ori))
                virtual_minutiae.append([x_i, y_i, ori]) #, distFromBg[y_i,x_i]
        virtual_minutiae = np.asarray(virtual_minutiae)
        if len(virtual_minutiae) > 3:
            virtual_des = descriptor.minutiae_descriptor_extraction(img, virtual_minutiae, self.patch_types, self.des_models,
                                                        self.patchIndexV,
                                                        batch_size=128)
            #show.show_minutiae(img,virtual_minutiae)
            texture_template = template.TextureTemplate(h=h, w=w,  minutiae=virtual_minutiae, des=virtual_des,
                                                  mask=mask)
            rolled_template.add_texture_template(texture_template)
        return rolled_template, enh_constrast_img

    def feature_extraction_single_rolled(self,img_file):
        block_size = 16

        if not os.path.exists(img_file):
            return None
        img = io.imread(img_file)
        h, w = img.shape
        mask = get_maps.get_quality_map_intensity(img)
        #if np.max(mask) == 0:
        #    print img_file
        #return None
        start = timeit.default_timer()
        mnt = self.minu_model.run_whole_image(img, minu_thr=0.2)
        stop = timeit.default_timer()
        minu_time = stop - start
        # show.show_minutiae(img, mnt, mask=mask, block=True, fname=None)
        # show.show_minutiae(img,mnt)

        # start = timeit.default_timer()
        des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models, self.patchIndexV,
                                                        batch_size=256, patch_size = self.patch_size)
        # stop = timeit.default_timer()
        # des_time = stop - start
        print minu_time#, des_time
        #texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)


        dir_map, _ = get_maps.get_maps_STFT(img, patch_size=64, block_size=block_size, preprocess=True)
        #stop = timeit.default_timer()

        blkH = h // block_size
        blkW = w // block_size
        # dir_map = np.zeros((blkH,blkW))
        # print stop - start
        #blkH, blkW = dir_map.shape



        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt, des=des, oimg=dir_map,
                                              mask=mask)

        rolled_template = template.Template()
        rolled_template.add_minu_template(minu_template)

        return rolled_template

    def feature_extraction(self,img_path,img_type='bmp',template_path=None, enhancement=False):


        img_files = glob.glob(img_path+'S*.'+img_type)
        assert(len(img_files)>0)

        img_files.sort()

        for i, img_file in enumerate(img_files):
            print img_file
            # if i<5:
            #     continue
            if i>7:
                break
                continue
                #break
            start = timeit.default_timer()
            img_name = os.path.basename(img_file)
            img_name = os.path.splitext(img_name)[0]
            if enhancement:
                rolled_template,enhanced_img = self.feature_extraction_single_rolled_enhancement(img_file)
                if template_path is not None:
                    enhanced_img = np.asarray(enhanced_img, dtype=np.uint8)
                    io.imsave(os.path.join(template_path, img_name + '.jpeg'), enhanced_img)
            else:
                rolled_template = self.feature_extraction_single_rolled(img_file)
            stop = timeit.default_timer()

            print stop - start
            if template_path is not None:
                fname = template_path + img_name + '.dat'
                template.Template2Bin_Byte_TF(fname, rolled_template, isLatent=False)

    def feature_extraction_MSP(self,img_path, N1=0,N2=10000, template_path=None,enhanced_img_path=None):

        assert(N2-N1>0)
        assert(template_path is not None)
        for i in range(N1,N2+1):
            #if i<10000:
            #    continue


            start = timeit.default_timer()
            img_file = os.path.join(img_path, str(i)+'.bmp')
            img_name = os.path.basename(img_file)
            fname = template_path + os.path.splitext(img_name)[0] + '.dat'
            if os.path.exists(fname):
                continue
            rolled_template = self.feature_extraction_single_rolled(img_file)
            stop = timeit.default_timer()
            if rolled_template is not None:
                template.Template2Bin_Byte_TF(fname, rolled_template, isLatent=True)
            print stop - start



def feature_extraction_single_latent(raw_img_file,AEC_img_file, mask_file, patch_types=None,des_models=None):
    ###
    #  input:
    # raw_img, original latent image
    # AEC_img, enhanced latent image by Autoencoder
    # mask:    ROI
    # main idea:
    # 1) Use AEC_img to estimate ridge flow and ridge spacing
    # 2) use AEC_image and raw_img to extract two different minutiae set
    ###
    raw_img = io.imread(raw_img_file)
    AEC_img = io.imread(AEC_img_file)
    mask = io.imread(mask_file)
    #mask = mask_dilation(mask, block_size=16)

    texture_img = preprocessing.FastCartoonTexture(raw_img, sigma=2.5, show=False)

    dir_map,fre_map,rec_img = get_maps.get_maps_STFT(AEC_img, patch_size=64, block_size=16, preprocess=True)

    descriptor_img = filtering.gabor_filtering_pixel(texture_img, dir_map+math.pi/2, fre_map, mask=mask, block_size=16, angle_inc=3)

    bin_img = binarization.binarization(texture_img, dir_map, block_size=16, mask=mask)

    enhanced_img = filtering.gabor_filtering_block(bin_img,dir_map+math.pi/2,fre_map,patch_size=64,block_size =16)
    enhanced_img = filtering.gabor_filtering_block(enhanced_img, dir_map+math.pi/2, fre_map, patch_size=64, block_size=16)

    # plt.subplot(131), plt.imshow(raw_img, cmap='gray')
    # plt.title('Input image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(descriptor_img, cmap='gray')
    # plt.title('Feature image'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(133), plt.imshow(enhanced_img, cmap='gray')
    # plt.title('Feature image'), plt.xticks([]), plt.yticks([])
    # plt.show(block=True)
    # plt.close()

    enhanced_AEC_img = filtering.gabor_filtering_block(AEC_img, dir_map + math.pi / 2, fre_map, patch_size=64,
                                                   block_size=16)
    bin_img = binarization.binarization(enhanced_AEC_img, dir_map, block_size=16, mask=mask)
    # plt.imshow(AEC_img,cmap='gray')
    # plt.show()
    # plt.close()

    bin_img2 = 1 - bin_img
    thin_img = skeletonize(bin_img2)
    # thin_img2 = thin_img.astype(np.uint8)
    # thin_img2[thin_img2 > 0] = 255

    mnt, thin_img2 = crossnumber.extract_minutiae(1 - thin_img, mask=mask, R=10)
    crossnumber.show_minutiae(thin_img, mnt)

    patchSize = 160
    oriNum = 64
    patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)
    if len(descriptor_img.shape)==2:
        h,w = descriptor_img.shape
        ret = np.empty((h, w, 3), dtype=np.float)
        ret[:, :, :] = descriptor_img[:, :, np.newaxis]
        descriptor_img = ret

    if len(enhanced_AEC_img.shape)==2:
        h,w = enhanced_AEC_img.shape
        ret = np.empty((h, w, 3), dtype=np.float)
        ret[:, :, :] = enhanced_AEC_img[:, :, np.newaxis]
        enhanced_AEC_img = ret

    des = descriptor.minutiae_descriptor_extraction(enhanced_AEC_img, mnt, patch_types, des_models, patchIndexV,batch_size=128)

    h,w = mask.shape
    blkH,blkW = dir_map.shape
    minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt, des=des, oimg=dir_map, mask=mask)

    latent_template = template.Template()
    latent_template.add_minu_template(minu_template)

    print des

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.',default='0')
    parser.add_argument('--N1', type=int,
        help='rolled index from which the enrollment starts', default=1)
    parser.add_argument('--N2', type=int,
                        help='rolled index from which the enrollment starts', default=100000)
    return parser.parse_args(argv)
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    des_model_dirs = []
    patch_types = []
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_1/20171206-093749/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(1)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_8/20171207-160445/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(8)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_11/20171207-143926/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(11)

    ### mobilenet
    # model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_2/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(2)
    # model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_8/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(8)
    model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_2/20180131-162542/'
    des_model_dirs.append(model_dir)
    patch_types.append(2)

    minu_model_dir = '/home/kaicao/Dropbox/Share/models/minutiae_AEC_128_fcn_aug2/'


    #LF_rolled = FeatureExtraction_Rolled()

    #

    dataset = 'NISTSD14'#NISTSD14'
    ## MSP background database
    if dataset == 'MSP':
        #template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP/version_efficient/'
        #img_path = '/media/kaicao/data2/Data/MSP_background/images/'
        #enhanced_img_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP/enhanced_image/'
        img_type = 'bmp'
        #template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP/version_2/'
        img_path = '/future/Data/Rolled/MSP/images/'
        template_path = '/future/Data/Rolled/MSP/template/'
        #des_model_dirs = None
        #minu_model_dir = None
        LF_rolled = FeatureExtraction_Rolled(patch_types=patch_types, des_model_dirs=des_model_dirs,
                                             minu_model_dir=minu_model_dir)
        #LF_rolled.feature_extraction(img_path,img_type=img_type)
        LF_rolled.feature_extraction_MSP(img_path,N1 = args.N1, N2 = args.N2, template_path=template_path)
    elif dataset == 'NISTSD27':
        img_path = '/media/kaicao/data2/Data/Rolled/NISTSD27/Image/'
        template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/NISTSD27/'

        LF_rolled = FeatureExtraction_Rolled(patch_types=patch_types, des_model_dirs=des_model_dirs,
                                             minu_model_dir=minu_model_dir)
        LF_rolled.feature_extraction(img_path, template_path=template_path,enhancement=True)
    #     LF_rolled.feature_extraction(img_path, img_type, template_path=template_path)
    elif dataset == 'NISTSD14':

        des_model_dirs = []
        patch_types = []
        model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_2/20180131-162542/'
        des_model_dirs.append(model_dir)
        patch_types.append(2)
        model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_8/20180131-163400/'
        des_model_dirs.append(model_dir)
        patch_types.append(8)
        model_dir = '/home/kaicao/Dropbox/Share/models/mobilenet/patch_type_11/20180131-214259/'
        des_model_dirs.append(model_dir)
        patch_types.append(11)

        # img_path = '/future/Data/Rolled/NISTSD14/Image2/'
        # template_path = '/future/Data/Rolled/NISTSD14/image_enhanced/'

        # for PRIP desktop
        img_path = '/home/kaicao/PRIP/Data/Rolled/NIST14/Image2/'
        template_path = '/media/kaicao/Seagate Expansion Drive/Research/AutomatedLatentRecognition/results/template/NISTSD14_test/'

        LF_rolled = FeatureExtraction_Rolled(patch_types=patch_types, des_model_dirs=des_model_dirs,
                                             minu_model_dir=minu_model_dir)
        LF_rolled.feature_extraction(img_path, template_path=template_path,enhancement=True)
    #     LF_rolled.feature_extraction(img_path, img_type, template_path=template_path)
