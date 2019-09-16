
import glob
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pathname = '/media/kaicao/Seagate Expansion Drive/Research/AutomatedLatentRecognition/selected_rolled_prints/'

pathname = '/media/kaicao/Data/AutomatedLatentRecognition/selected_rolled_prints/'
t_pathname = 'media/kaicao/Data/AutomatedLatentRecognition/Patches/'


def extract_patches(pathname,t_pathname):
    subjects = glob.glob(pathname+'*')

    n = 0
    step = 64
    patch_size = 128
    for subject in subjects:
        imgfiles = glob.glob(subject + '/high*.bmp')
        #img = scipy.misc.imread(imgfiles[0])
        img = mpimg.imread(imgfiles[0])
        #imgplot = plt.imshow(img,cmap='gray')
        #plt.show()
        #print img
        h,w = img.shape
        for i in range(step*2,h-patch_size*2,step):
            for j in range(step*2,w-patch_size*2,step):
                patch = img[i:i+patch_size,j:j+patch_size]
                #print patch

                #imgplot = plt.imshow(patch, cmap='gray')
                if n==50:
                    print patch
                    print n
                #mpimg.imsave(t_pathname+str(n)+'.jpg',patch)
                scipy.misc.imsave(t_pathname+str(n)+'.jpeg', patch)
                n = n + 1
                print n
                #if n>100000:
                #    return


if __name__ == "__main__":
    extract_patches(pathname, t_pathname)
