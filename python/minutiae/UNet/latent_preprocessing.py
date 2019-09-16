import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import skimage.io
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
import math
import cv2
import sklearn.linear_model

def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))

def local_constrast_enhancement(img):
    h,w = img.shape
    img = img.astype(np.float32)
    mean = cv2.GaussianBlur(img,(15,15),0)
    normalized = img - mean
    var = abs(normalized)

    var = cv2.GaussianBlur(var, (15, 15), 0)

    normalized = normalized/(var+0.0001) *0.75

    normalized = np.clip(normalized,-1,1)
    return normalized

def LowpassFiltering(img,L):

    h,w = img.shape
    h2,w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2-h, 0, w2-w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    #img_fft = fftshift(fft2(img,h2,w2));

    img_fft = img_fft * L

    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h,:w]
    #plt.imshow(rec_img, cmap='gray')
    #plt.show()
    return rec_img


def compute_gradient_norm(input):
    input = input.astype(np.float32)
    h,w = input.shape

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    #input = double(input);
    #[Gx, Gy] = gradient(input);
    #out = sqrt(Gx.*Gx + Gy.*Gy);
    return out
def construct_dictionary():
    oriNum = 30

    patch_size2 = 16
    patch_size = 32

    X, Y = np.meshgrid(range(-patch_size2,patch_size2), range(-patch_size2,patch_size2))
    dict = []
    s = []
    for spacing in range(6,13):
        for valley_spacing in range(3,spacing//2):
            ridge_spacing = spacing - valley_spacing
            for k in range(oriNum):
                theta = k * np.pi / oriNum
                X_r = X * np.cos(theta) - Y * np.sin(theta)
                for offset in range(spacing-1):
                    X_r_offset = X_r + offset + ridge_spacing / 2
                    X_r_offset = np.remainder(X_r_offset, spacing)
                    Y1 = np.zeros((patch_size,patch_size))
                    Y2 = np.zeros((patch_size,patch_size))

                    Y1[X_r_offset <= ridge_spacing] = X_r_offset[X_r_offset <= ridge_spacing]
                    Y2[X_r_offset > ridge_spacing] = X_r_offset[X_r_offset > ridge_spacing] - ridge_spacing
                    element = -np.sin(2 * math.pi * (Y1 / ridge_spacing / 2)) + np.sin(2 * math.pi * (Y2 / valley_spacing / 2))
                    #plt.imshow(element,cmap='gray')
                    #plt.show()
                    dict.append(element.reshape(patch_size*patch_size,))
                    s.append(spacing)
    for i in range(len(dict)):
        dict[i] = dict[i]/np.linalg.norm(dict[i])
        #print np.linalg.norm(dict[i])
    dict = np.array(dict)
    s = np.array(s)
    return dict,s

def code_step(X, D,n_nonzero_coefs=6):
  model = sklearn.linear_model.OrthogonalMatchingPursuit(
          n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False, normalize=False
  )
  #C = sklearn.
  model.fit(D.T, X.T)
  return model.coef_


def denoising_using_dictonary(img, dict):
    # img: input latent image
    # dict: ridge structure dictionary constructed by function construct_dictionary()
    nrof_elements, nrof_pixels = dict.shape
    patch_size = int(np.sqrt(nrof_pixels))
    block_size = 16

    h, w = img.shape
    blkH = (h - patch_size) / block_size + 1
    blkW = (w - patch_size) / block_size + 1

    rec_img = np.zeros((h,w))
    patches = []
    for i in range(0, blkH):
        print i
        for j in range(0, blkW):
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()
            patch = patch-np.mean(patch)
            patch = np.reshape(patch,(nrof_pixels,))
            patches.append(patch)
            # patch = np.concatenate((patch,patch),axis=1)
            # patch = patch - np.mean(patch)
            # coef = code_step(patch, dict)
            # rec_patch = np.dot(dict.T,coef)
            # rec_patch = np.reshape(rec_patch,(patch_size,patch_size))
            # rec_img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size] += rec_patch
            # plt.subplot(121), plt.imshow(np.reshape(patch,(patch_size,patch_size)), cmap='gray')
            # plt.title('Input patch'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            # plt.title('reconstructed patch'), plt.xticks([]), plt.yticks([])
            # plt.show()
    patches = np.array(patches)
    coef = code_step(patches, dict)
    rec_patches = np.dot(coef, dict)
    n = 0
    for i in range(0, blkH):
        print i
        for j in range(0, blkW):
            rec_patch = np.reshape(rec_patches[n], (patch_size, patch_size))
            n+=1
            rec_img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size] += rec_patch

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input patch'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(rec_img, cmap='gray')
    plt.title('reconstructed patch'), plt.xticks([]), plt.yticks([])
    plt.show()
def FastCartoonTexture(img,sigma=2.5,show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(-FFTsize / 2, FFTsize / 2), range(-FFTsize / 2, FFTsize / 2))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r/FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma)** 4)
    img_low = LowpassFiltering(img, L)

    gradim1=  compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1,L)

    gradim2=  compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2,L)

    diff = gradim1-gradim2
    ar1 = np.abs(gradim1)
    diff[ar1>1] = diff[ar1>1]/ar1[ar1>1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff-cmin)/(cmax-cmin)
    weight[diff<cmin] = 0
    weight[diff>cmax] = 1


    u = weight * img_low + (1-weight)* img

    temp = img - u

    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)

    temp1[temp1 < 0] = 0
    temp1[temp1 >255] = 255
    v = temp1
    if show:
        plt.imshow(v,cmap='gray')
        plt.show()
    return u, v

def STFT(img):
    patch_size = 64
    block_size = 16
    ovp_size = (patch_size-block_size)//2
    h0, w0 = img.shape
    img = cv2.copyMakeBorder(img, ovp_size, ovp_size, ovp_size, ovp_size, cv2.BORDER_CONSTANT, value=0)

    h,w = img.shape
    blkH = (h - patch_size)//block_size
    blkW = (w - patch_size)//block_size

    #-------------------------
    # Bandpass filter
    # -------------------------
    RMIN = 3  # min allowable ridge spacing
    RMAX = 18 # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN

    x, y = np.meshgrid(range(-patch_size / 2,patch_size / 2), range(-patch_size / 2,patch_size / 2))
    r = np.sqrt(x*x + y*y) + 0.0001

    dRLow = 1. / (1 + (r / FHIGH)**6) # low pass     butterworth     filter
    dRHigh = 1. / (1 + (FLOW / r)**6) # high    pass     butterworth     filter
    dBPass = dRLow * dRHigh  # bandpass

    sigma = patch_size/3
    weight = np.exp(-(x*x + y*y)/(sigma*sigma))
    rec_img = np.zeros((h,w))
    for i in range(0,blkH):
        for j in range(0,blkW):
            patch =img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size].copy()
            f = np.fft.fft2(patch)
            fshift = np.fft.fftshift(f)
            #magnitude_spectrum = np.log(np.abs(fshift))

            filtered = dBPass*fshift
            norm = np.linalg.norm(filtered)
            filtered = filtered/norm
            f_ifft = np.fft.ifftshift(filtered)#*np.sqrt(np.abs(fshift)))
            rec_patch = np.real(np.fft.ifft2(f_ifft))
            rec_img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size] += rec_patch*weight
            # plt.subplot(221), plt.imshow(patch, cmap='gray')
            # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
            # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            # plt.subplot(223), plt.imshow(rec_patch, cmap='gray')
            # plt.title('reconstructed patch'), plt.xticks([]), plt.yticks([])
            # plt.subplot(224), plt.imshow(dBPass, cmap='gray')
            # plt.title('reconstructed patch'), plt.xticks([]), plt.yticks([])
            # plt.show()


    rec_img = rec_img[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]

    rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img) - np.min(rec_img)) * 255

    img = img[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(rec_img, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


    return rec_img




def test_local_equalize(imgfiles):
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)
    for imgfile in imgfiles:

        img = skimage.io.imread(imgfile)
        # estimate the noise standard deviation from the noisy image
        sigma_est = np.mean(estimate_sigma(img, multichannel=True))
        print("estimated noise standard deviation = {}".format(sigma_est))



        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 6),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})

        # Equalization
        selem = disk(30)
        img_eq = rank.equalize(img, selem=selem)

        # slow algorithm
        denoise = denoise_nl_means(img_eq, h=1.15 * sigma_est, fast_mode=False,
                                   **patch_kw)

        # fast algorithm
        denoise_fast = denoise_nl_means(img_eq, h=0.8 * sigma_est, fast_mode=True,
                                        **patch_kw)


        ax[0].imshow(img,cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('noisy')
        ax[1].imshow(denoise,cmap='gray')
        ax[1].axis('off')
        ax[1].set_title('non-local means\n(slow)')
        ax[2].imshow(img_eq,cmap='gray')
        ax[2].axis('off')
        ax[2].set_title('local equalize')


        fig.tight_layout()
        plt.show()
        print img.shape

def test_filtering_regional_maxima(imgfiles):
    for imgfile in imgfiles:

        img = skimage.io.imread(imgfile)
        image = gaussian_filter(img, 1)

        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.min()
        mask = image

        dilated = reconstruction(seed, mask, method='dilation')
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                            ncols=3,
                                            figsize=(8, 2.5),
                                            sharex=True,
                                            sharey=True)

        ax0.imshow(image, cmap='gray')
        ax0.set_title('original image')
        ax0.axis('off')
        ax0.set_adjustable('box-forced')

        ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
        ax1.set_title('dilated')
        ax1.axis('off')
        ax1.set_adjustable('box-forced')

        ax2.imshow(image - dilated, cmap='gray')
        ax2.set_title('image - dilated')
        ax2.axis('off')
        ax2.set_adjustable('box-forced')

        fig.tight_layout()
        plt.show()

if __name__=='__main__':
    dict, spacing = construct_dictionary()

    imgfiles = glob.glob('/home/kaicao/Dropbox/Research/Data/Latent/NISTSD27/image/*.bmp')
    imgfiles.sort()
    for imgfile in imgfiles:
        img = cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img,np.float32)
        denoising_using_dictonary(img, dict)
        #STFT(img)
        u,texture = FastCartoonTexture(img)
        STFT(texture)
    #test_filtering_regional_maxima(imgfiles)