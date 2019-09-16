from skimage import data, io
from skimage.morphology import square
from skimage.transform import rescale, resize
import glob
import matplotlib.pylab as plt
import scipy.ndimage
from skimage.morphology import skeletonize, square, dilation
from skimage.filters import gabor_kernel, gaussian
#import minutiae
import binarization
import get_maps
import math
import preprocessing
import numpy as np
import timeit
#from numba import jit

def gabor_filtering_block(img,dir_map,fre_map,mask = None,patch_size=64,block_size =16):


    img = img.astype(np.float)
    ovp_size = (patch_size - block_size) // 2
    h0, w0 = img.shape
    if mask is None:
        mask = np.ones((h0,w0),dtype=np.int)
    img = np.lib.pad(img, (ovp_size, ovp_size), 'symmetric')
    # img = cv2.copyMakeBorder(img, ovp_size, ovp_size, ovp_size, ovp_size, cv2.BORDER_CONSTANT, value=0)

    x, y = np.meshgrid(range(-patch_size / 2, patch_size / 2), range(-patch_size / 2, patch_size / 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    sigma = patch_size / 3

    weight = np.exp(-(x * x + y * y) / (sigma * sigma))

    h,w = img.shape
    blkH = (h - patch_size)//block_size + 1
    blkW = (w - patch_size)//block_size + 1
    rec_img = np.zeros((h, w))
    for i in range(0, blkH):
        for j in range(0, blkW):

            if mask[i * block_size+block_size//2, j * block_size+block_size//2] == 0:
                continue
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()
            patch -= np.mean(patch)
            patch_FFT = np.fft.fft2(patch)
            #patch_FFT = np.fft.fftshift(patch_FFT)
            if fre_map[i,j]<0:
                continue
            kernel = gabor_kernel(fre_map[i,j], theta=dir_map[i,j], sigma_x=4, sigma_y=4)
            f = kernel.real
            f = f - np.mean(f)
            f = f/(np.linalg.norm(f)+0.00001)
            kernel_FFT = np.fft.fft2(f, (patch_size,patch_size))
            #kernel_FFT = np.fft.fftshift(kernel_FFT)
            patch_FFT = patch_FFT * kernel_FFT
            #patch_IFFT = np.fft.ifftshift(patch_FFT)  # *np.sqrt(np.abs(fshift)))
            rec_patch = np.real(np.fft.ifft2(patch_FFT))
            rec_patch = np.fft.ifftshift(rec_patch)
            plt.subplot(121), plt.imshow(patch, cmap='gray')
            plt.title('Input mage'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('Enhanced image'), plt.xticks([]), plt.yticks([])
            plt.show(block=True)
            plt.close()
            #if weight is not None:
            rec_patch = rec_patch * weight


            rec_img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size] += rec_patch

    rec_img = rec_img[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]

    rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img) - np.min(rec_img)) * 255

    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Input mage'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(rec_img, cmap='gray')
    # plt.title('Enhanced image'), plt.xticks([]), plt.yticks([])


    return rec_img


def get_gabor_filters(angle_inc = 3,fre_num=30):
    ori_num = 180 // angle_inc
    gaborfilter = np.zeros((ori_num,fre_num),dtype=object)
    for i in range(ori_num):
        ori = i * angle_inc / 180.0 * math.pi
        for j in range(fre_num):
            if j<5:
                continue

            kernel = gabor_kernel(j*0.01, theta=ori, sigma_x=4, sigma_y=4)
            kernel = kernel.real

            # plt.imshow(kernel,cmap='gray')
            # plt.show()
            # plt.close()
            kernel = kernel - np.mean(kernel)
            norm = np.linalg.norm(kernel)

            kernel = kernel/(norm+0.00001)
            gaborfilter[i, j] = kernel.real

    return gaborfilter

def gabor_filtering_pixel(img,dir_map,fre_map,mask = None, block_size =16,angle_inc=3,gabor_filters = None):
    # block_size is the block size of the dir_map and fre_map
    h, w = img.shape
    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    if block_size > 1:
        cos2Theta = np.cos(dir_map * 2)
        sin2Theta = np.sin(dir_map * 2)
        cos2Theta = scipy.ndimage.interpolation.zoom(cos2Theta, block_size)
        sin2Theta = scipy.ndimage.interpolation.zoom(sin2Theta, block_size)
        frequency = scipy.ndimage.interpolation.zoom(fre_map, block_size)
        angle = np.arctan2(sin2Theta, cos2Theta) * 0.5
    else:
        angle = dir_map
        frequency = fre_map
    angle = angle / math.pi * 180
    angle = angle.astype(int)
    angle[angle < 0] = angle[angle < 0] + 180
    angle[angle == 180] = 0
    angle_ind = angle//angle_inc

    #angle_ind = np.zeros((h,w),dtype=int)+8
    #frequency_ind = np.zeros((h,w),dtype=int)+12
    frequency_ind = np.around(frequency*100).astype(int)
    # get gabor filters

    if gabor_filters is None:
        gabor_filters = get_gabor_filters()

    img = img.astype(np.float)
    h,w = img.shape
    enh_img = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0,   w):
            if mask[i,j] == 0:
                continue

            frequency_ind_b = frequency_ind[i,j]
            angle_ind_b = angle_ind[i,j]

            if frequency_ind_b<5 or frequency_ind_b>=30:
                continue
            kernel = gabor_filters[angle_ind_b,frequency_ind_b]
            kh,kw = kernel.shape
            sh = kh//2
            sw = kw//2
            if i<sh or j<sw or i>h-sh-1 or  j>w-sh-1:
                continue
            block = img[i-sh:i+sh+1,j-sw:j+sw+1]
            # t = block*kernel
            # if np.sum(t)>100 or np.sum(t)<-100:
            #     print t
            enh_img[i,j] = np.sum(block*kernel)

    enh_img = (enh_img-np.min(enh_img))/(np.max(enh_img)-np.min(enh_img)+0.0001)*255
    return enh_img




# if __name__ == '__main__':
#     enhanced_img_path = '../../../../Share/enhanced_latents_3/'
#     enhanced_img_files = glob.glob(enhanced_img_path + '*.jpeg')
#     enhanced_img_files.sort()
#
#     img_path = '../../../../Share/enhanced_latents_3/'
#     imgfiles = glob.glob(img_path + '*.jpeg')
#     imgfiles.sort()
#
#     mask_path = '../../../Data/Latent/NISTSD27/maskNIST27/'
#     maskfiles = glob.glob(mask_path + '*.bmp')
#     maskfiles.sort()
#
#     block_size = 1
#     for i, imgfile in enumerate(imgfiles):
#         if i<1:
#             continue
#         raw_img = io.imread(imgfile)
#         enhanced_img = io.imread(enhanced_img_files[i])
#         texture_img = preprocessing.FastCartoonTexture(raw_img,sigma=2.5,show=False)
#         maskfile = maskfiles[i]
#         mask = io.imread(maskfile)
#
#         mask = mask_dilation(mask, block_size=16)
#
#         #dir_map = get_maps.get_ridge_flow_gradient(img, mask=mask, block_size=block_size, sigma=5)
#         dir_map,fre_map,rec_img = get_maps.get_maps_STFT(enhanced_img, patch_size=64, block_size=16, preprocessing=True)
#         feature_img = get_maps.gabor_filtering_pixel(texture_img, dir_map, fre_map, mask=mask, block_size=16, angle_inc=3)
#
#         plt.subplot(121), plt.imshow(raw_img, cmap='gray')
#         plt.title('Input image'), plt.xticks([]), plt.yticks([])
#         plt.subplot(122), plt.imshow(feature_img, cmap='gray')
#         plt.title('Feature image'), plt.xticks([]), plt.yticks([])
#
#         plt.show()
#
#         bin_img = binarization.binarization(img, dir_map, block_size=16, mask=mask)
#         # plt.imshow(bin_img,cmap='gray')
#         # plt.show()
#         # plt.close()
#
#         bin_img2 = 1 - bin_img
#         thin_img = skeletonize(bin_img2)
#         # thin_img2 = thin_img.astype(np.uint8)
#         # thin_img2[thin_img2 > 0] = 255
#
#         mnt, thin_img2 = minutiae.extract_minutiae(1 - thin_img, mask=mask, R =10)
#         minutiae.show_minutiae(thin_img, mnt)


if __name__=='__main__':
    get_gabor_filters(angle_inc=3, fre_num=30)
