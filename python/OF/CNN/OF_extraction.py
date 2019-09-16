import numpy as np
import tensorflow as tf
import scipy.io as sio
from skimage import data, io
import sys
import os
import glob
import matplotlib.pylab as plt
sys.path.append('../../')
import load
import show
import preprocessing

class ImportGraph():
    def __init__(self, model_dir,OF_file):
        # create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.OF_center = sio.loadmat(OF_file)['center']

        # load orientation field center
        center_num, dim = self.OF_center.shape
        cos2Theta = []
        sin2Theta = []
        ind = ['1', '10',  '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115',  '116', \
         '117','118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '13',  '14', '15', '16', '17', '18', '19', '2', '20', \
         '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42',
         '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66',\
         '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82',  '83', '84', '85', '86', '87','88', '89',\
         '9',   '90','91', '92', '93','94', '95', '96', '97', '98', '99']

        #img = np.ones((160, 160))
        for i in range(center_num):
            k = int(ind[i])-1
            cospart = self.OF_center[k, :dim / 2]
            sinpart = self.OF_center[k, dim / 2:]
            cospart = cospart.reshape((10, 10))
            sinpart = sinpart.reshape((10, 10))
            #cospart = np.transpose(cospart)
            #sinpart = np.transpose(sinpart)
            #dir_map = np.arctan2(sinpart, cospart) * 0.5
            #show.show_orientation_field(img, dir_map)
            cos2Theta.append(cospart)
            sin2Theta.append(sinpart)

        self.cos2Theta = cos2Theta
        self.sin2Theta = sin2Theta

        with self.graph.as_default():
            meta_file, ckpt_file = load.get_model_filenames(os.path.expanduser(model_dir))
            model_dir_exp = os.path.expanduser(model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("batch_join:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.ph,self.pw = self.images_placeholder.get_shape().as_list()[1:3]
            self.logits_placeholder = tf.get_default_graph().get_tensor_by_name('mul:0')

    def run(self, img,mask=None,  stride = 32, block_size = 16):
        #feed_dict = {self.images_placeholder: img}
        #minutiae_cylinder = self.sess.run(self.minutiae_cylinder_placeholder, feed_dict=feed_dict)

        h,w = img.shape[:2]
        if len(img.shape) == 2:
            h, w = img.shape
            ret = np.empty((h, w, 3), dtype=np.uint8)
            ret[:, :, :] = img[:, :, np.newaxis]
            img = ret

        if mask is None:
            mask = np.ones((h,w),dtype=np.int)
        mask[mask>1] = 1

        blkH = h//block_size
        blkW = w//block_size

        cos2Theta = np.zeros((blkH,blkW))
        sin2Theta = np.zeros((blkH,blkW))

        patch_size = self.ph

        patchH = (h - patch_size)//stride + 1
        patchW = (w - patch_size)//stride + 1

        patches = []
        loc_x = []
        loc_y = []
        step = stride//block_size
        for i in range(patchH):
            y = np.int(i*stride)
            for j in range(patchW):
                x = np.int(j*stride)

                if np.sum(mask[y:y+patch_size,x:x+patch_size])>patch_size*patch_size/2:
                    patch =img[y:y+patch_size,x:x+patch_size,:]
                    if patch.shape[0]<160:
                        print patch.shape
                    patches.append(patch)
                    loc_x.append(j*step)
                    loc_y.append(i*step)
        if len(patches) == 0:
            dir_map = None
            return dir_map

        patches = np.asarray(patches)

        feed_dict = {self.images_placeholder: patches,self.phase_train_placeholder:False}
        logits = self.sess.run(self.logits_placeholder, feed_dict=feed_dict)

        for i in range(len(patches)):
            selected = np.argmax(logits[i,:])
            cos2Theta[loc_y[i]:loc_y[i]+10,loc_x[i]:loc_x[i]+10] +=self.cos2Theta[selected]
            sin2Theta[loc_y[i]:loc_y[i]+10,loc_x[i]:loc_x[i]+10] +=self.sin2Theta[selected]
        dir_map = np.arctan2(sin2Theta,cos2Theta)*0.5
        show.show_orientation_field(img,dir_map=dir_map)
        return None



    # def minutiae_extraction(self,img):
    #     h, w = img.shape
    #     x = []
    #     y = []
    #     weight = get_weights(128,128,12)
    #     nrof_samples = len(range(0, h, opt.SHAPE // 2)) * len(range(0, w, opt.SHAPE // 2))
    #     patches = np.zeros((nrof_samples, opt.SHAPE, opt.SHAPE, 1))
    #     n = 0
    #     for i in range(0, h - opt.SHAPE + 1, opt.SHAPE // 2):
    #
    #         for j in range(0, w - opt.SHAPE + 1, opt.SHAPE // 2):
    #             print j
    #             patch = img[i:i + opt.SHAPE, j:j + opt.SHAPE, np.newaxis]
    #             x.append(j)
    #             y.append(i)
    #             patches[n, :, :, :] = patch
    #             n = n + 1
    #             # print x[-1]
    #     minutiae_cylinder_array = self.run(patches)
    #     minutiae_cylinder = np.zeros((h, w, 12))
    #     minutiae_cylinder_array[:, -10:, :, :] = 0
    #     minutiae_cylinder_array[:, :10, :, :] = 0
    #     minutiae_cylinder_array[:, :, -10:, :] = 0
    #     minutiae_cylinder_array[:, :, 10, :] = 0
    #
    #     for i in range(n):
    #         minutiae_cylinder[y[i]:y[i] + opt.SHAPE, x[i]:x[i] + opt.SHAPE, :] = minutiae_cylinder[
    #                                                                              y[i]:y[i] + opt.SHAPE,
    #                                                                              x[i]:x[i] + opt.SHAPE, :] + \
    #                                                                              minutiae_cylinder_array[i] * weight
    #     # print minutiae_cylinder
    #     minutiae = prepare_data.get_minutiae_from_cylinder(minutiae_cylinder, thr=0.05)
    #
    #     # cv2.imwrite('test_0.jpeg', (minutiae_cylinder[:, :, 0:3]) * 255)
    #     # cv2.imwrite('test_1.jpeg', (minutiae_cylinder[:, :, 3:6]) * 255)
    #     # cv2.imwrite('test_2.jpeg', (minutiae_cylinder[:, :, 6:9]) * 255)
    #     # cv2.imwrite('test_3.jpeg', (minutiae_cylinder[:, :, 9:12]) * 255)
    #     # prepare_data.show_features(img, minutiae, fname=os.path.basename(file)[:-4] +'.jpeg')
    #
    #     minutiae = prepare_data.refine_minutiae(minutiae, dist_thr=10, ori_dist=np.pi / 4)
    #
    #     return minutiae

if __name__ == '__main__':
    model_dir = '/media/kaicao/data2/AutomatedLatentRecognition/models/OF/facenet/20171229-120921/'
    OF_center_file = 'OriCenter.mat'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    img_path = '../../../../../Data/Latent/NISTSD27/image/'

    mask_path = '../../../../../Data/Latent/NISTSD27/maskNIST27/'

    img_files = glob.glob(img_path+'*.bmp')
    img_files.sort()

    mask_files = glob.glob(mask_path+'*.bmp')
    mask_files.sort()




    for i in range(250, len(img_files)):
        img = io.imread(img_files[i])
        mask = io.imread(mask_files[i])
        img = preprocessing.FastCartoonTexture(img)
        img[mask == 0] = 0

        model = ImportGraph(model_dir,OF_center_file)
        dir_map = model.run(img,mask=mask)
