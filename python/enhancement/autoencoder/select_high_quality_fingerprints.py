
import glob
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copyfile
import os.path

pathname = '/media/kaicao/Data/AutomatedLatentRecognition/selected_rolled_prints/'
t_pathname = '/media/kaicao/Data/AutomatedLatentRecognition/high_quality_images/'


def select_images(pathname,t_pathname):
    subjects = glob.glob(pathname+'*')

    n = 0
    step = 64
    patch_size = 128
    for subject in subjects:
        imgfiles = glob.glob(subject + '/high*.bmp')
        img = scipy.misc.imread(imgfiles[0])
	t = os.path.basename(imgfiles[0])
	#\copyfile(imgfiles[0],t_pathname+t)
	#print t
        #img = mpimg.imread(imgfiles[0])
        #imgplot = plt.imshow(img,cmap='gray')
        #plt.show()
        #print img
        #h,w = img.shape
        #for i in range(step*2,h-patch_size*2,step):
        scipy.misc.imsave(t_pathname+t[:-3]+'jpeg',img)
if __name__ == "__main__":
    select_images(pathname, t_pathname)
