from tensorpack.dataflow import *
from tensorpack.utils.globvars import globalns as opt
import glob
import numpy as np
from tensorpack import (Trainer, QueueInput,
                        ModelDescBase, dataflow, StagingInputWrapper,
                        MultiGPUTrainerBase,
                        TowerContext)
#from AutoEncoder2_denoising import *


opt.SHAPE = 128
opt.BATCH = 128

datadir = '/home/kaicao/Dropbox/Research/AutomatedLatentRecognition/Data/minutiae_cylinder'


class ImageFromFile_AutoEcoder(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        #self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle

    def size(self):
        return len(self.files)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for f in self.files:
            matrix = np.load(f)
            #matrix = np.float32(matrix)
            h,w,c = matrix.shape
            #im = matrix[:opt.SHAPE, :opt.SHAPE, 0:1]#np.squeeze(matrix[:,:,0])
            #cylinder = matrix[:opt.SHAPE, :opt.SHAPE, 2::]
            #mx = np.random.randint(w - opt.SHAPE)
            #my = np.random.randint(h - opt.SHAPE)
            #im = matrix[my:my+opt.SHAPE, mx:mx+opt.SHAPE, 0:1]  # matrix[:opt.SHAPE, :opt.SHAPE, 0:1]#np.squeeze(matrix[:,:,0])
            #cylinder = matrix[my:my+opt.SHAPE, mx:mx+opt.SHAPE, 2::]  # matrix[:opt.SHAPE, :opt.SHAPE, 2::]
            im = matrix[:, :, 0:1]  # np.squeeze(matrix[:,:,0])
            cylinder = matrix[:, :, 2::]
            # if self.channel == 3:
            #     im = im[:, :, ::-1]
            # if self.resize is not None:
            #     im = cv2.resize(im, tuple(self.resize[::-1]))
            # if self.channel == 1:
            #     im = im[:, :, np.newaxis]
            yield [im,cylinder]



imgs = glob.glob(datadir + '/*.npy')  # outfile = data_path + subjectID + '_latent.npy'
ds = ImageFromFile_AutoEcoder(imgs, channel=1, shuffle=True)
# augmentor = get_augmentors()
#ds = MultiThreadMapData(
#     ds, nr_thread=1,
#    map_func=lambda dp: [augmentor.augment(dp[0]), dp[1]], buffer_size=1000)
#ds = AugmentImageComponent(ds, get_augmentors())

ds = PrefetchDataZMQ(ds, nr_proc = 1)
dftools.dump_dataflow_to_lmdb(ds, '/home/kaicao/minutiae_cylinder.lmdb')

#ds = BatchData(ds, opt.BATCH)
#TestDataSpeed(ds,size=10000).start()