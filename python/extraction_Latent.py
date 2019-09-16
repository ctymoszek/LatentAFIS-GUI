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
import cv2
import scipy.spatial.distance
import get_maps
import preprocessing
import enhancement.filtering, enhancement.binarization
#import minutiae.crossnumber
import descriptor
import descriptor.CNN.evaluation.descriptor as descriptor
import os
import template
import minutiae.UNet.minutiae_AEC as minutiae_AEC
import show
import enhancement.autoencoder.enhancement_AEC
import ROI.RCNN

from timeit import default_timer as timer
class FeatureExtraction_Latent:
    def __init__(self,patch_types=None,des_model_dirs=None,minu_model_dir=None, enhancement_model_dir = None, ROI_model_dir=None):
        self.des_models = None
        self.patch_types = patch_types
        self.minu_model = None
        self.minu_model_dir = minu_model_dir
        self.des_model_dirs = des_model_dirs
        self.enhancement_model_dir = enhancement_model_dir
        self.ROI_model_dir = ROI_model_dir
        self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=60)

        if self.minu_model_dir is not None:
            self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))

        patchSize = 160
        oriNum = 64
        self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

        if self.des_model_dirs is not None:
            self.des_models = []
            for model_dir in self.des_model_dirs:
                self.des_models.append(descriptor.ImportGraph(des_model_dirs))
        if self.enhancement_model_dir is not None:
            self.enhancement_model = enhancement_AEC.ImportGraph(enhancement_model_dir)
        if self.ROI_model_dir is not None:
            self.ROI_model = (RCNN.ImportGraph(ROI_model_dir))
            # self.minu_models = []
            # for model_dir in self.des_model_dirs:
            #     self.minu_models.append(descriptor.ImportGraph(model_dir))

    def feature_extraction_single_latent_evaluation(self,img_file, mask_file, AEC_img_file,output_path = None ):

        img = io.imread(img_file)
        name = os.path.basename(img_file)
        AEC_img = io.imread(AEC_img_file)
        mask = io.imread(mask_file)
        h,w = mask.shape
        #mask = mask_dilation(mask, block_size=16)
        latent_template = template.Template()
        block = False
        minu_thr = 0.3

        contrast_img = preprocessing.local_constrast_enhancement(img)
        # Two ways for orientation field estimation
        #  Use the AEC_img and STFT on texture image
        dir_map_sets = []
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
        dir_map, fre_map = get_maps.get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)
        dir_map_sets.append(dir_map)

        blkH, blkW = dir_map.shape

        dir_map, fre_map = get_maps.get_maps_STFT(AEC_img, patch_size=64, block_size=16, preprocess=True)
        dir_map_sets.append(dir_map)

        #dir_map, fre_map = get_maps.get_maps_STFT(contrast_img, patch_size=64, block_size=16, preprocess=True)
        #dir_map_sets.append(dir_map)

        # based on the OF, we can use texture image and AEC image for frequency field estimation

        fre_map_sets = []
        quality_map, fre_map = get_maps.get_quality_map_ori_dict(AEC_img, self.dict, self.spacing, dir_map=dir_map_sets[0],
                                                                 block_size=16)
        fre_map_sets.append(fre_map)

        quality_map, fre_map = get_maps.get_quality_map_ori_dict(contrast_img, self.dict, self.spacing,
                                                                 dir_map=dir_map_sets[1],
                                                                 block_size=16)
        fre_map_sets.append(fre_map)

        descriptor_imgs = [texture_img]
        descriptor_imgs.append(contrast_img)
        enh_texture_img = filtering.gabor_filtering_pixel(texture_img, dir_map + math.pi / 2, fre_map_sets[0], mask=mask,
                                                         block_size=16, angle_inc=3)
        descriptor_imgs.append(enh_texture_img)
        enh_contrast_img = filtering.gabor_filtering_pixel(contrast_img, dir_map + math.pi / 2, fre_map_sets[1], mask=mask,
                                                           block_size=16, angle_inc=3)
        descriptor_imgs.append(enh_contrast_img)

        minutiae_sets = []
        mnt = self.minu_model.run(texture_img, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_texture_img.jpeg'
        show.show_minutiae(texture_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(contrast_img, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_contrast_img.jpeg'
        show.show_minutiae(contrast_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(enh_texture_img, minu_thr=minu_thr)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_enh_texture_img.jpeg'
        show.show_minutiae(enh_texture_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(enh_contrast_img, minu_thr=minu_thr)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_enh_contrast_img.jpeg'
        show.show_minutiae(enh_contrast_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(AEC_img, minu_thr=minu_thr)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)

        fname = output_path + os.path.splitext(name)[0] + '_AEC_img.jpeg'
        show.show_minutiae(AEC_img, mnt, block=block, fname=fname)



        for mnt in minutiae_sets:
            for des_img in descriptor_imgs:
                des = descriptor.minutiae_descriptor_extraction(des_img, mnt, self.patch_types, self.des_models,
                                                        self.patchIndexV,batch_size=128)
                minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt,
                                                  des=des, oimg=dir_map_sets[1], mask=mask)
                latent_template.add_minu_template(minu_template)

        return latent_template

    def feature_extraction_single_latent_evaluation_AEM18T(self,img_file, mask_file, AEC_img_file,output_path = None ):

        img = io.imread(img_file)
        name = os.path.basename(img_file)
        AEC_img = io.imread(AEC_img_file)
        mask_CNN = io.imread(mask_file)
        h,w = mask_CNN.shape
        #mask = mask_dilation(mask, block_size=16)
        latent_template = template.Template()
        block = False
        minu_thr = 0.3

        # template set 1: no ROI and enhancement are required
        # texture image is used for coase segmentation

        descriptor_imgs = []
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
        descriptor_imgs.append(texture_img)
        contrast_img_mean = preprocessing.local_constrast_enhancement(img)
        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)

        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(texture_img, self.dict_all, self.dict_ori,
                                                                      self.dict_spacing, block_size=16, process=False)
        quality_map_pixel = cv2.resize(quality_map, (0, 0), fx=16, fy=16)
        mask_coarse = quality_map_pixel > 0.3
        mask_coarse = mask_coarse.astype(np.int)
        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(AEC_img, self.dict_all, self.dict_ori,self.dict_spacing, block_size=16, process=False)


        minutiae_sets = []
        mnt = self.minu_model.run(texture_img, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask_coarse)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_texture_img.jpeg'
        show.show_minutiae(texture_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(contrast_img_mean, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask_coarse)
        minutiae_sets.append(mnt)

        fname = output_path + os.path.splitext(name)[0] + '_contrast_img_mean.jpeg'
        show.show_minutiae(contrast_img_mean, mnt, block=block, fname=fname)


        mnt = self.minu_model.run(contrast_img_guassian, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask_coarse)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_contrast_img_guassian.jpeg'
        show.show_minutiae(contrast_img_guassian, mnt, block=block, fname=fname)

        #show.show_orientation_field(AEC_img,dir_map)

        enh_texture_img = filtering.gabor_filtering_pixel(texture_img, dir_map + math.pi / 2, fre_map,
                                                          mask=np.ones((h, w), np.int),
                                                          block_size=16, angle_inc=3)

        descriptor_imgs.append(enh_texture_img)

        enh_constrast_img = filtering.gabor_filtering_pixel(contrast_img_guassian, dir_map + math.pi / 2, fre_map,
                                                          mask=np.ones((h,w),np.int),
                                                          block_size=16, angle_inc=3)

        descriptor_imgs.append(enh_constrast_img)

        quality_map2, _ , _ = get_maps.get_quality_map_dict(enh_texture_img, self.dict_all,self.dict_ori,self.dict_spacing, block_size=16,
                                                                      process=False)
        quality_map_pixel2 = cv2.resize(quality_map2, (0, 0), fx=16, fy=16)

        mask = quality_map_pixel2 > 0.55


        mask = mask.astype(np.int)
        mask = mask_coarse * mask
        mask = mask * mask_CNN

        mnt = self.minu_model.run(AEC_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_AEC_img.jpeg'
        show.show_minutiae(AEC_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(enh_texture_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)

        fname = output_path + os.path.splitext(name)[0] + '_enh_texture_img.jpeg'
        show.show_minutiae(enh_texture_img, mnt, block=block, fname=fname)

        mnt = self.minu_model.run(enh_constrast_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        fname = output_path + os.path.splitext(name)[0] + '_enh_constrast_img.jpeg'
        show.show_minutiae(enh_constrast_img, mnt, block=block, fname=fname)


        blkH, blkW = dir_map.shape
        for mnt in minutiae_sets:
            for des_img in descriptor_imgs:
                des = descriptor.minutiae_descriptor_extraction(des_img, mnt, self.patch_types, self.des_models,
                                                        self.patchIndexV, batch_size=128)
                minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt,
                                                  des=des, oimg=dir_map, mask=mask)
                latent_template.add_minu_template(minu_template)

        return latent_template

    def feature_extraction_single_latent_demo(self,img_file, output_path = None ):
        block = True
        img = io.imread(img_file)

        plt.imshow(img, cmap='gray')
        plt.show(block=block)
        plt.close()



        name = os.path.basename(img_file)

        #AEC_img = io.imread(AEC_img_file)
        #mask_CNN = io.imread(mask_file)
        mask_CNN,_ = self.ROI_model.run(img)
        h,w = mask_CNN.shape
        #mask = mask_dilation(mask, block_size=16)
        latent_template = template.Template()

        minu_thr = 0.3

        # template set 1: no ROI and enhancement are required
        # texture image is used for coase segmentation

        descriptor_imgs = []
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)




        descriptor_imgs.append(texture_img)
        contrast_img_mean = preprocessing.local_constrast_enhancement(img)
        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)

        quality_map, _, _ = get_maps.get_quality_map_dict(texture_img, self.dict_all, self.dict_ori,
                                                                      self.dict_spacing, block_size=16, process=False)
        quality_map_pixel = cv2.resize(quality_map, (0, 0), fx=16, fy=16)
        plt.imshow(quality_map_pixel,cmap='gray')
        plt.show()
        mask_coarse = quality_map_pixel > 0.3
        mask_coarse = mask_coarse.astype(np.int)
        mask = mask_coarse * mask_CNN
        show.show_mask(mask_CNN, img, fname='mask_RCNN.jpeg',block=block)
        show.show_mask(mask_coarse,img,fname = 'mask_coarse.jpeg',block=block)
        show.show_mask(mask, img, fname='mask.jpeg',block=block)




        #show.show_mask(mask, AEC_img, fname='mask_AEC.jpeg',block=block)
        # plt.imshow(AEC_img,cmap = 'gray')
        # plt.show(block=block)
        # plt.close()



        show.show_mask(mask_CNN, img, fname='mask_RCNN.jpeg',block=block)

        # AEC_img[mask == 0] = 128
        # plt.imshow(AEC_img, cmap='gray')
        # plt.show(block=block)
        # plt.close()

        AEC_img = self.enhancement_model.run(texture_img)
        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(AEC_img, self.dict_all, self.dict_ori,self.dict_spacing, block_size=16, process=False)


        show.show_orientation_field(img, dir_map,mask = mask,fname='OF.jpeg')




        # mnt = self.minu_model.run(contrast_img_mean, minu_thr=0.1)
        # mnt = self.remove_spurious_minutiae(mnt, mask)
        # minutiae_sets.append(mnt)
        #
        # fname = output_path + os.path.splitext(name)[0] + '_contrast_img_mean.jpeg'
        # show.show_minutiae(contrast_img_mean, mnt, block=block, fname=fname)

        start = timer()
        enh_contrast_img = filtering.gabor_filtering_pixel(contrast_img_guassian, dir_map + math.pi / 2, fre_map,
                                                          mask=mask,
                                                          block_size=16, angle_inc=3)

        bin_img = binarization.binarization(enh_contrast_img,dir_map,mask=mask)
        thin_img = skeletonize(bin_img)
        plt.imshow(thin_img,cmap='gray')
        plt.show(block=True)
        dt = timer() - start
        print(dt)
        show.show_image(texture_img, mask=mask, block=True, fname='cropped_texture_image.jpeg')
        show.show_image(AEC_img, mask=mask, block=True, fname='cropped_AEC_image.jpeg')
        show.show_image(enh_contrast_img, mask=mask, block=True, fname='cropped_enh_image.jpeg')

        #np.ones((h, w), np.int)
        descriptor_imgs.append(enh_contrast_img)


        quality_map2, _ , _ = get_maps.get_quality_map_dict(enh_contrast_img, self.dict_all,self.dict_ori,self.dict_spacing, block_size=16,
                                                                      process=False)
        quality_map_pixel2 = cv2.resize(quality_map2, (0, 0), fx=16, fy=16)

        mask2 = quality_map_pixel2 > 0.55

        minutiae_sets = []
        mnt = self.minu_model.run(texture_img, minu_thr=0.1)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)
        fname = 'minutiae_texture_img.jpeg'
        show.show_minutiae(texture_img, mnt, mask=mask,block=block, fname=fname)

        mnt = self.minu_model.run(AEC_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)
        fname = 'minutiae_AEC_img.jpeg'
        show.show_minutiae(AEC_img, mnt, mask=mask, block=block, fname=fname)

        mnt = self.minu_model.run(enh_contrast_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)

        fname = 'minutiae_enh_contrast_img.jpeg'
        show.show_minutiae(img, mnt, mask=np.ones((h,w)), block=block, fname=fname)
        #show.show_minutiae(img, mnt, mask=mask,block=block, fname=fname)

        fname = 'minutiae_skeleton.jpeg'
        show.show_minutiae(img, mnt, mask=mask, block=block, fname=fname)
        #
        # mnt = self.minu_model.run(enh_constrast_img, minu_thr=0.3)
        # mnt = self.remove_spurious_minutiae(mnt, mask)
        # minutiae_sets.append(mnt)
        # fname = output_path + os.path.splitext(name)[0] + '_enh_constrast_img.jpeg'
        # show.show_minutiae(enh_constrast_img, mnt, block=block, fname=fname)


        # blkH, blkW = dir_map.shape
        # for mnt in minutiae_sets:
        #     for des_img in descriptor_imgs:
        #         des = descriptor.minutiae_descriptor_extraction(des_img, mnt, self.patch_types, self.des_models,
        #                                                 self.patchIndexV, batch_size=128)
        #         minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt,
        #                                           des=des, oimg=dir_map, mask=mask)
        #         latent_template.add_minu_template(minu_template)

        return None

    def feature_extraction_single_latent(self,img_file, output_path = None, show_processes=False ):
        #block = True
        img = io.imread(img_file)

        name = os.path.basename(img_file)
        mask_CNN,_ = self.ROI_model.run(img)
        h,w = mask_CNN.shape
        #mask = mask_dilation(mask, block_size=16)
        latent_template = template.Template()

        minu_thr = 0.3

        # template set 1: no ROI and enhancement are required
        # texture image is used for coase segmentation

        descriptor_imgs = []
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)




        descriptor_imgs.append(texture_img)

        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)

        quality_map, _, _ = get_maps.get_quality_map_dict(texture_img, self.dict_all, self.dict_ori,
                                                                      self.dict_spacing, block_size=16, process=False)
        quality_map_pixel = cv2.resize(quality_map, (0, 0), fx=16, fy=16)
        #plt.imshow(quality_map_pixel,cmap='gray')
        #plt.show()
        mask_coarse = quality_map_pixel > 0.3
        mask_coarse = mask_coarse.astype(np.int)
        mask = mask_coarse * mask_CNN
        # show.show_mask(mask_CNN, img, fname='mask_RCNN.jpeg',block=block)
        # show.show_mask(mask_coarse,img,fname = 'mask_coarse.jpeg',block=block)
        # show.show_mask(mask, img, fname='mask.jpeg',block=block)




        #show.show_mask(mask, AEC_img, fname='mask_AEC.jpeg',block=block)
        # plt.imshow(AEC_img,cmap = 'gray')
        # plt.show(block=block)
        # plt.close()



        #show.show_mask(mask_CNN, img, fname='mask_RCNN.jpeg',block=block)

        # AEC_img[mask == 0] = 128
        # plt.imshow(AEC_img, cmap='gray')
        # plt.show(block=block)
        # plt.close()

        AEC_img = self.enhancement_model.run(texture_img)
        quality_map, dir_map, fre_map = get_maps.get_quality_map_dict(AEC_img, self.dict_all, self.dict_ori,self.dict_spacing, block_size=16, process=False)

        blkH, blkW = dir_map.shape

        if show_processes:
            show.show_orientation_field(img, dir_map,mask = mask,fname='OF.jpeg')




        # mnt = self.minu_model.run(contrast_img_mean, minu_thr=0.1)
        # mnt = self.remove_spurious_minutiae(mnt, mask)
        # minutiae_sets.append(mnt)
        #
        # fname = output_path + os.path.splitext(name)[0] + '_contrast_img_mean.jpeg'
        # show.show_minutiae(contrast_img_mean, mnt, block=block, fname=fname)


        enh_contrast_img = filtering.gabor_filtering_pixel(contrast_img_guassian, dir_map + math.pi / 2, fre_map,
                                                          mask=mask,
                                                          block_size=16, angle_inc=3)

        enh_texture_img = filtering.gabor_filtering_pixel(texture_img, dir_map + math.pi / 2, fre_map,
                                                          mask=mask,
                                                          block_size=16, angle_inc=3)

        if show_processes:
            show.show_image(texture_img, mask=mask, block=True, fname='cropped_texture_image.jpeg')
            show.show_image(AEC_img, mask=mask, block=True, fname='cropped_AEC_image.jpeg')
            show.show_image(enh_contrast_img, mask=mask, block=True, fname='cropped_enh_image.jpeg')

        #np.ones((h, w), np.int)
        descriptor_imgs.append(enh_contrast_img)


        quality_map2, _ , _ = get_maps.get_quality_map_dict(enh_contrast_img, self.dict_all,self.dict_ori,self.dict_spacing, block_size=16,
                                                                      process=False)
        quality_map_pixel2 = cv2.resize(quality_map2, (0, 0), fx=16, fy=16)

        mask2 = quality_map_pixel2 > 0.50

        #mask = mask*mask2

        minutiae_sets = []
        mnt = self.minu_model.run(contrast_img_guassian, minu_thr=0.05)
        mnt = self.remove_spurious_minutiae(mnt, mask)
        minutiae_sets.append(mnt)
        if show_processes:
            fname = 'minutiae_texture_img.jpeg'
            show.show_minutiae(texture_img, mnt, mask=mask,block=block, fname=fname)

        mnt = self.minu_model.run(AEC_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)
        if show_processes:
            fname = 'minutiae_AEC_img.jpeg'
            show.show_minutiae(AEC_img, mnt, mask=mask, block=block, fname=fname)

        mnt = self.minu_model.run(enh_contrast_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)
        if show_processes:
            fname = 'minutiae_enh_contrast_img.jpeg'
            show.show_minutiae(enh_contrast_img, mnt, mask=mask,block=block, fname=fname)

        mnt = self.minu_model.run(enh_texture_img, minu_thr=0.3)
        mnt = self.remove_spurious_minutiae(mnt, mask2)
        minutiae_sets.append(mnt)

        # minutiae template 1
        des = descriptor.minutiae_descriptor_extraction(texture_img, minutiae_sets[0], self.patch_types, self.des_models,
                                                         self.patchIndexV, batch_size=128)

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=minutiae_sets[0],
                                                   des=des, oimg=dir_map, mask=mask)
        latent_template.add_minu_template(minu_template)

        # minutiae template 2
        des = descriptor.minutiae_descriptor_extraction(texture_img, minutiae_sets[1], self.patch_types,
                                                        self.des_models,
                                                        self.patchIndexV, batch_size=128)

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=minutiae_sets[1],
                                              des=des, oimg=dir_map, mask=mask)
        latent_template.add_minu_template(minu_template)

        # minutiae template 3
        des = descriptor.minutiae_descriptor_extraction(enh_texture_img, minutiae_sets[2], self.patch_types,
                                                        self.des_models,
                                                        self.patchIndexV, batch_size=128)

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=minutiae_sets[2],
                                              des=des, oimg=dir_map, mask=mask)
        latent_template.add_minu_template(minu_template)

        # minutiae template 4
        des = descriptor.minutiae_descriptor_extraction(enh_texture_img, minutiae_sets[3], self.patch_types,
                                                        self.des_models,
                                                        self.patchIndexV, batch_size=128)

        minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=minutiae_sets[3],
                                              des=des, oimg=dir_map, mask=mask)
        latent_template.add_minu_template(minu_template)



        return latent_template


    def remove_spurious_minutiae(self,mnt,mask):

        minu_num = mnt.shape[0]
        flag = np.ones((minu_num,), np.uint8)
        h,w = mask.shape[:2]
        R = 5
        for i in range(minu_num):
            x = mnt[i,0]
            y = mnt[i,1]
            x = np.int(x)
            y = np.int(y)
            if x<R or y<R or x>w-R-1 or y>h-R-1:
                flag[i] = 0
            if mask[y-R,x-R]==0 or mask[y-R,x+R]==0 or mask[y+R,x-R]==0 or mask[y+R,x+R]==0:
                flag[i] = 0
        mnt = mnt[flag>0,:]
        return mnt


    def feature_extraction(self,img_path,template_path=None):

        img_files = glob.glob(img_path+'*.bmp')
        assert(len(img_files)>0)


        img_files.sort()


        for i, img_file in enumerate(img_files):
            if i<11:
                continue
            start = timeit.default_timer()
            #latent_template = self.feature_extraction_single_latent(img_file,output_path=template_path)
            latent_template = self.feature_extraction_single_latent_demo(img_file, output_path=template_path)

            stop = timeit.default_timer()

            print (stop - start)
            if template_path is not None and rolled_template is not None:
                img_name = os.path.basename(img_file)
                fname = template_path + os.path.splitext(img_name)[0]+'.dat'
                template.Template2Bin_Byte_TF(fname, latent_template, isLatent=True)


def demo_minutiae_extraction(img_path,minu_model_dir):

    img_files = glob.glob(img_path+'*.bmp')
    img_files.sort()
    minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))
    block = True
    for i, img_file in enumerate(img_files):
        if i<11:
            continue
        img = io.imread(img_file)
        name = os.path.basename(img_file)
        h, w = img.shape
        mask = np.ones((h,w),dtype=np.uint8)

        minu_thr = 0.3

        texture_img = preprocessing.FastCartoonTexture(img)
        contrast_img = preprocessing.local_constrast_enhancement_gaussian(img)

        dir_map, fre_map = get_maps.get_maps_STFT(contrast_img, patch_size=64, block_size=16, preprocess=True)

        dict, spacing,_ = get_maps.construct_dictionary(ori_num=60)
        quality_map, fre_map = get_maps.get_quality_map_ori_dict(contrast_img, dict, spacing,
                                                                 dir_map=dir_map,
                                                                 block_size=16)
        enh_texture_img = filtering.gabor_filtering_pixel(contrast_img, dir_map + math.pi / 2, fre_map,
                                                          mask=mask,
                                                          block_size=16, angle_inc=3)

        mnt = minu_model.run(contrast_img, minu_thr=0.1)
        #mnt = minu_model.remove_spurious_minutiae(mnt, mask)
        #minutiae_sets.append(mnt)
        #fname = output_path + os.path.splitext(name)[0] + '_texture_img.jpeg'
        show.show_minutiae(contrast_img, mnt, block=block, fname=None)

        mnt = minu_model.run(texture_img, minu_thr=0.1)
        # mnt = minu_model.remove_spurious_minutiae(mnt, mask)
        # minutiae_sets.append(mnt)
        # fname = output_path + os.path.splitext(name)[0] + '_texture_img.jpeg'
        show.show_minutiae(texture_img, mnt, block=block, fname=None)

        print(i)

def ExtractLatent(img_file):
    # args = parse_arguments(sys.argv[1:])
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    # minutiae descriptor models
    des_model_dirs = []
    patch_types = []
    dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(dir, '/scratch/LatentAFIS/models/')
    des_model_dirs.append(model_dir)
    patch_types.append(1)
    des_model_dirs.append(os.path.join(model_dir, 'facenet_raw_enhanced_type_1/20171206-093749/'))
    #des_model_dirs.append(model_dir)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_8/20171207-160445/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(8)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_11/20171207-143926/'
    # des_model_dirs.append(model_dir)
    # patch_types.append(11)

    # minutiae extraction model
    minu_model_dir = os.path.join(model_dir, 'minutiae_AEC_128_fcn_aug2/')

    # enhancement model
    enhancement_model_dir = os.path.join(model_dir, 'enhancement/')

    # ROI model
    ROI_model_dir = os.path.join(model_dir, 'ROI/15_12_0140.h5')

    # = '../../../Data/Latent/NISTSD27/image/'
    #mask_path =  '../../../Data/Latent/NISTSD27/maskNIST27/'

    #demo_minutiae_extraction(img_path, minu_model_dir)

    template_path = os.path.join(dir, 'Data/current_latent_data/')

    LF_Latent = FeatureExtraction_Latent(patch_types=patch_types, des_model_dirs=des_model_dirs[1],
                                          minu_model_dir=minu_model_dir,enhancement_model_dir=enhancement_model_dir,ROI_model_dir=ROI_model_dir)
    LF_Latent.feature_extraction_single(img_file, template_path=template_path)
	
    demo_minutiae_extraction(img_file, minu_model_dir)
	
    return
