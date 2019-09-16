from skimage import data, io
import numpy as np
import glob
import matplotlib.pylab as plt
from skimage.morphology import skeletonize, square, dilation
import math
from skimage.morphology import square
from skimage.transform import rescale, resize
import sys
sys.path.append('OF')
sys.path.append('descriptor/CNN/evaluation')
sys.path.append('enhancement')
sys.path.append('minutiae')
sys.path.append('utils')
sys.path.append('minutiae/Minutiae_UNet/')
import get_maps
import preprocessing
import filtering, binarization
import crossnumber
import descriptor
import os
import template
import minutiae_AEC

def mask_dilation(mask,block_size = 16):
    print block_size
    blk_mask = mask[block_size//2::block_size,block_size//2::block_size]
    blk_mask = dilation(blk_mask, square(2))
    mask = rescale(blk_mask,block_size)
    mask[mask>0] = 1
    return mask
    # plt.imshow(mask,cmap='gray')
    # plt.show()

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
    mask = mask_dilation(mask, block_size=16)

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
if __name__ == '__main__':
    # args = parse_arguments(sys.argv[1:])
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    AEC_img_path = '../../../../Share/enhanced_latents_3/'
    AEC_img_file = glob.glob(AEC_img_path + '*.jpeg')
    AEC_img_file.sort()

    img_path = '../../../Data/Latent/NISTSD27/image/'
    imgfiles = glob.glob(img_path + '*.bmp')
    imgfiles.sort()

    mask_path = '../../../Data/Latent/NISTSD27/maskNIST27/'
    mask_files = glob.glob(mask_path + '*.bmp')
    mask_files.sort()

    minu_models = []
    minu_models =minutiae_AEC.ImportGraph('/home/kaicao/Dropbox/Share/models/minutiae_AEC_128_fcn_aug2/')

    model_dirs = []
    patch_types = []
    model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_1/20171206-093749/'
    model_dirs.append(model_dir)
    patch_types.append(1)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_8/20171207-160445/'
    # model_dirs.append(model_dir)
    # patch_types.append(8)
    # model_dir = '/home/kaicao/Dropbox/Share/models/facenet_raw_enhanced_type_11/20171207-143926/'
    # model_dirs.append(model_dir)
    # patch_types.append(11)

    models = []
    for model_dir in model_dirs:
        models.append(descriptor.ImportGraph(model_dir))


    block_size = 1
    for i, imgfile in enumerate(imgfiles):
        if i<0:
            continue
        feature_extraction_single_latent(imgfiles[i], AEC_img_file[i], mask_files[i], patch_types = patch_types,des_models = models)
        minutiae_extraction_latent(args.load, args.sample_dir, imgs, block=False)