import RCNN
import os
import skimage.io as io
import matplotlib.pyplot as plt
#os.environ["KERAS_BACKEND"]= 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



mymodel = RCNN.load_model('/home/kaicao/Research/Mask_framework/15_12_0140.h5')

img = plt.imread('/home/kaicao/Dropbox/Research/Data/Latent/NISTSD27/image/001.bmp')
mask, mask_ori = func_demo.generate_mask(mymodel, img,
              '/home/kaicao/Dropbox/Research/Data/Latent/NISTSD27',
              gen_type = 'box', fuse_thres = 1)

print mask.shape
plt.set_cmap('gray')
plt.imshow(mask)
plt.show()
