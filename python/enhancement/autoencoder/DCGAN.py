#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import glob
import numpy as np
import os, sys
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf
from tensorpack.tfutils.common import get_tensors_by_names
from GAN import GANTrainer, RandomZData, GANModelDesc
import scipy.misc
import pdb
import timeit

"""
1. Download the 'aligned&cropped' version of CelebA dataset
   from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. Start training:
    ./DCGAN-CelebA.py --data /path/to/img_align_celeba/ --crop-size 140
    Generated samples will be available through tensorboard

3. Visualize samples with an existing model:
    ./DCGAN-CelebA.py --load path/to/model --sample

You can also train on other images (just use any directory of jpg files in
`--data`). But you may need to change the preprocessing.

A pretrained model on CelebA is at https://drive.google.com/open?id=0B9IPQTvr2BBkLUF2M0RXU1NYSkE
"""

# global vars
opt.SHAPE = 512
opt.BATCH = 32 
opt.Z_DIM = 512 


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]

    def generator(self, z):
        """ return an image generated from z"""
        nf = 16
        l = FullyConnected('fc0', z, nf * 64 * 4 * 4, nl=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 64])
        l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, [8, 8, nf * 32])
            l = Deconv2D('deconv2', l, [16, 16, nf * 16])
            l = Deconv2D('deconv3', l, [32, 32, nf*8])
            l = Deconv2D('deconv4', l, [64, 64, nf * 4])
            l = Deconv2D('deconv5', l, [128, 128, nf * 2])
            l = Deconv2D('deconv6', l, [256, 256, nf * 1])
            l = Deconv2D('deconv7', l, [512, 512, 3], nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        """ return a (b, 1) logits"""
        nf = 16
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', nf * 16)
                 .BatchNorm('bn4').LeakyReLU()
                 .Conv2D('conv5', nf * 32)
                 .BatchNorm('bn5').LeakyReLU()
                 .Conv2D('conv6', nf * 64)
                 .BatchNorm('bn6').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random_uniform([opt.BATCH, opt.Z_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_augmentors():
    augs = []
    if opt.load_size:
        augs.append(imgaug.Resize(opt.load_size))
    if opt.crop_size:
        augs.append(imgaug.CenterCrop(opt.crop_size))
    augs.append(imgaug.Resize(opt.SHAPE))
    return augs


def get_data(datadir):
    imgs = glob.glob(datadir + '/*.jpeg')
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 5)
    return ds


def sample(model, model_path,sample_path, num = 100, output_name='gen/gen'):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    batch_size = 100
    #print range(num/batch_size)
    #pdb.set_trace()
    for i in range(1):
        pred = SimpleDatasetPredictor(pred, RandomZData((batch_size, opt.Z_DIM)))
        n=0
        for o in pred.get_result():
            #pdb.set_trace()
            o, zs = o[0] + 1, o[1]

            o = o * 128.0
            o = np.clip(o, 0, 255)
            o = o[:, :, :, ::-1]
            #viz = stack_patches(o, nr_row=10, nr_col=10, viz=True)
            #
            for j in range(batch_size):
                n = n + 1
                pdb.set_trace()
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                print img.shape
                scipy.misc.imsave('%s%09d.jpeg' % (sample_path,n), img)
            print n
            if n>=num:
                break

def sample2(model, model_path,sample_path, num = 100, output_name='gen/gen'):
    config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    graph = config._maybe_create_graph()
    batch_size = 250
    n = 0
    with graph.as_default():
        input = PlaceholderInput()
        input.setup(config.model.get_inputs_desc())
        with TowerContext('', is_training=False):
            config.model.build_graph(input)

        input_tensors = get_tensors_by_names(config.input_names)
        output_tensors = get_tensors_by_names(config.output_names)

        sess = config.session_creator.create_session()
        config.session_init.init(sess)
        if sess is None:
            sess = tf.get_default_session()
        for i in  range(num/batch_size):
            #dp = RandomZData((batch_size, opt.Z_DIM))
	    

            start = timeit.default_timer()
            dp = [np.random.normal(-1, 1, size=(batch_size, opt.Z_DIM))]
            #dp = tf.placeholder_with_default(dp, [None, G.Z_DIM], name='z')
            feed = dict(zip(input_tensors, dp))

            output = sess.run(output_tensors, feed_dict=feed)
            

            
            o, zs = output[0] + 1, output[1]
            for j in range(len(o)):
                n = n + 1
                #pdb.set_trace()
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                #pdb.set_trace()
                #print img.shape
                scipy.misc.imsave('%s%09d.jpeg' % (sample_path,n), img)
            print n
            stop = timeit.default_timer()
            print stop - start 
        #return output



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument('--log_dir', help='directory to save checkout point', type=str, default='/media/kaicao/Data/checkpoint/FingerprintSynthesis/tensorpack/DCGAN/')
    args = parser.parse_args()
    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.sample:
        sample(Model(), args.load)
    else:
        assert args.data
        #logger.auto_set_dir()
        logger.set_logger_dir(args.log_dir)
        config = TrainConfig(
            model=Model(),
            callbacks=[ModelSaver()],
            dataflow=get_data(args.data),
            steps_per_epoch=1000,
            max_epoch=300,
            session_init=SaverRestore(args.load) if args.load else None
        )
        GANTrainer(config).train()
