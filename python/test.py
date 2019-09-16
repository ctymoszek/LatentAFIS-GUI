import numpy as np
import scipy.ndimage
import matplotlib.pylab as plt
import glob
import orientation_field as OF

if __name__=='__main__':

    pathname = '/home/kaicao/Dropbox/Research/Data/Latent/NISTSD27/image/'

    imgfiles = glob.glob(pathname + '*.bmp')
    imgfiles.sort()

    for imgfile in imgfiles:
        img = scipy.ndimage.imread(imgfile)
        OF.STFT(img)
        plt.imshow(img,cmap='gray')
        plt.show()

