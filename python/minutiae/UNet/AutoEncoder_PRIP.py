#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
import argparse
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf
#from tensorpack.base import RNGDataFlow
#from GAN import GANTrainer, RandomZData, GANModelDesc
import scipy.misc
from tensorpack import Trainer
from tensorpack import *
from tensorpack.utils.globvars import globalns as opt
import glob, os
import cv2
from tensorpack import (Trainer, QueueInput,
                        ModelDescBase, DataFlow, StagingInputWrapper,
                        MultiGPUTrainerBase,
                        TowerContext)
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized

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
opt.SHAPE = 128
opt.BATCH = 32



class ImageFromFile_AutoEcoder_Prediction(RNGDataFlow):
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
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
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
            im = cv2.imread(f, self.imread_mode)
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                #im = im[200:328, 200:328, np.newaxis]
                im = im[:, :, np.newaxis]
            h,w,c = im.shape
            im.astype(float)
            im = im/ 128.0 - 1
            for i in xrange(0,h-opt.SHAPE,opt.SHAPE//2):
                for j in xrange(0,w-opt.SHAPE,opt.SHAPE//2):
                    patch = im[i:i+opt.SHAPE,i:i+opt.SHAPE,:]
                    yield patch
            #mx = np.random.randint(w-opt.SHAPE)
            #my = np.random.randint(h-opt.SHAPE)
            #im = im[my:my+opt.SHAPE,mx:mx+opt.SHAPE,:]
            #yield [im]

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
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
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
            matrix = np.float32(matrix)
            h,w,c = matrix.shape
            #im = matrix[:opt.SHAPE, :opt.SHAPE, 0:1]#np.squeeze(matrix[:,:,0])
            #cylinder = matrix[:opt.SHAPE, :opt.SHAPE, 2::]
            mx = np.random.randint(w - opt.SHAPE)
            my = np.random.randint(h - opt.SHAPE)
            im = matrix[my:my+opt.SHAPE, mx:mx+opt.SHAPE, 0:1]  # matrix[:opt.SHAPE, :opt.SHAPE, 0:1]#np.squeeze(matrix[:,:,0])
            cylinder = matrix[my:my+opt.SHAPE, mx:mx+opt.SHAPE, 2::]  # matrix[:opt.SHAPE, :opt.SHAPE, 2::]

            # if self.channel == 3:
            #     im = im[:, :, ::-1]
            # if self.resize is not None:
            #     im = cv2.resize(im, tuple(self.resize[::-1]))
            # if self.channel == 1:
            #     im = im[:, :, np.newaxis]
            yield [im,cylinder]


class AEC_Model(ModelDesc):
    # # replace BatchNorm by LayerNorm
    # @auto_reuse_variable_scope
    # def discriminator(self, imgs):
    #     nf = 64
    #     with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
    #             argscope(LeakyReLU, alpha=0.2):
    #         l = (LinearWrap(imgs)
    #              .Conv2D('conv0', nf, nl=LeakyReLU)
    #              .Conv2D('conv1', nf * 2)
    #              .LayerNorm('ln1').LeakyReLU()
    #              .Conv2D('conv2', nf * 4)
    #              .LayerNorm('ln2').LeakyReLU()
    #              .Conv2D('conv3', nf * 8)
    #              .LayerNorm('ln3').LeakyReLU()
    #              .FullyConnected('fct', 1, nl=tf.identity)())
    #     return tf.reshape(l, [-1])

    def _get_inputs(self):
        #return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]
                #InputDesc(tf.int32, (None, opt.SHAPE, opt.SHAPE, 3), 'label')]
        return [InputDesc(tf.float32, (None, None, None, 1), 'input'),
         InputDesc(tf.float32, (None, None, None, 12), 'label')]


    def collect_variables(self, scope='reconstruction'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assert self.vars


    @auto_reuse_variable_scope
    def reconstruction(self, imgs):
        """ return a (b, 1) logits"""
        nf = 64
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
             l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU) #64
                 .Conv2D('conv1', nf * 2)           #32
                 .LayerNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)           #16
                 .LayerNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)           #8
                 .LayerNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', nf * 8)           #4
                 .LayerNorm('bn4').LeakyReLU()
                 .Conv2D('conv5', nf * 8)           #2
                 .LayerNorm('bn5').LeakyReLU()
                 .Conv2D('conv6', nf * 8)           #1
                 .LayerNorm('bn6').LeakyReLU()())
             l = tf.tanh(l, name='feature')
             l = Dropout(l)
        #l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, nf * 8)
            l = Deconv2D('deconv2', l, nf * 8)
            l = Deconv2D('deconv3', l, nf * 8)
            l = Deconv2D('deconv4', l, nf * 4)
            l = Deconv2D('deconv5', l, nf * 2)
            l = Deconv2D('deconv6', l, nf * 1)
            l = Deconv2D('deconv7', l, 12, nl=tf.identity)
            #l = LeakyReLU(l, alpha=0.2, name='gen')
            l = tf.nn.relu(l, name='gen')
        return l
    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1
        target = inputs[1]/256.0
        #image_label = inputs[1]
        #image_label = image_label / 128.0 - 1

        #z = tf.random_normal([opt.BATCH, opt.Z_DIM], name='z_train')
        #z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('reconstruction'):
                prediction = self.reconstruction(image_pos)
            #tf.summary.image('generated-samples', image_gen, max_outputs=30)

            #with tf.variable_scope('reconstruction'):
            #    prediction = self.generator(image_pos)

        self.cost = tf.nn.l2_loss(prediction - target, name="L2loss")
        # the Wasserstein-GAN losses
        # self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        # self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        #
        # # the gradient penalty loss
        # gradients = tf.gradients(vec_interp, [interp])[0]
        # gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        # gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        # gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        # add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)
        #
        # self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)
        add_moving_summary(self.cost)
        tf.summary.image('original', image_pos, max_outputs=30)
        tf.summary.image('prediction', prediction[:,:,:,0:3], max_outputs=30)
        tf.summary.image('target', target[:,:,:,0:3], max_outputs=30)

        self.build_losses()
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt

    def build_losses(self):
        """D and G play two-player minimax game with value function V(G,D)

          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
        """
        with tf.name_scope("L2_loss"):
            self.loss = self.cost
            #add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)

    #@memoized
    def get_optimizer(self):
        return self._get_optimizer()


class Cao_Model(ModelDesc):
    # # replace BatchNorm by LayerNorm
    # @auto_reuse_variable_scope
    # def discriminator(self, imgs):
    #     nf = 64
    #     with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
    #             argscope(LeakyReLU, alpha=0.2):
    #         l = (LinearWrap(imgs)
    #              .Conv2D('conv0', nf, nl=LeakyReLU)
    #              .Conv2D('conv1', nf * 2)
    #              .LayerNorm('ln1').LeakyReLU()
    #              .Conv2D('conv2', nf * 4)
    #              .LayerNorm('ln2').LeakyReLU()
    #              .Conv2D('conv3', nf * 8)
    #              .LayerNorm('ln3').LeakyReLU()
    #              .FullyConnected('fct', 1, nl=tf.identity)())
    #     return tf.reshape(l, [-1])

    def _get_inputs(self):
        #return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]
                #InputDesc(tf.int32, (None, opt.SHAPE, opt.SHAPE, 3), 'label')]
        return [InputDesc(tf.float32, (None, None, None, 1), 'input'),
         InputDesc(tf.float32, (None, None, None, 12), 'label')]


    def collect_variables(self, scope='reconstruction'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assert self.vars


    @auto_reuse_variable_scope
    def reconstruction(self, imgs):
        """ return a (b, 1) logits"""
        nf = 32
        with argscope(Conv2D, kernel_shape=3, stride=1), \
             argscope(LeakyReLU, alpha=0.2):
            # nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
            # encoder
            e1 = Conv2D('conv1', imgs, nf, nl=LeakyReLU)
            e2 = Conv2D('conv2', e1, nf * 2)
            e3 = Conv2D('conv3', e2, nf * 4)
            e4 = Conv2D('conv4', e3, nf * 4)
            e5 = Conv2D('conv5', e4, nf * 4)
            e6 = Conv2D('conv6', e5, nf * 2)
            e7 = Conv2D('conv7', e6, nf * 1)
            l = Conv2D('conv8', e7, 12, nl=tf.identity)
            l = tf.nn.relu(l, name='gen')
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1
        target = inputs[1]/256.0
        #image_label = inputs[1]
        #image_label = image_label / 128.0 - 1

        #z = tf.random_normal([opt.BATCH, opt.Z_DIM], name='z_train')
        #z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('reconstruction'):
                prediction = self.reconstruction(image_pos)
            #tf.summary.image('generated-samples', image_gen, max_outputs=30)

            #with tf.variable_scope('reconstruction'):
            #    prediction = self.generator(image_pos)

        self.cost = tf.nn.l2_loss(prediction - target, name="L2loss")
        # the Wasserstein-GAN losses
        # self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        # self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        #
        # # the gradient penalty loss
        # gradients = tf.gradients(vec_interp, [interp])[0]
        # gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        # gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        # gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        # add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)
        #
        # self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)
        add_moving_summary(self.cost)
        tf.summary.image('original', image_pos, max_outputs=30)
        tf.summary.image('prediction', prediction[:,:,:,0:3], max_outputs=30)
        tf.summary.image('target', target[:,:,:,0:3], max_outputs=30)

        self.build_losses()
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt

    def build_losses(self):
        """D and G play two-player minimax game with value function V(G,D)

          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
        """
        with tf.name_scope("L2_loss"):
            self.loss = self.cost
            #add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)

    #@memoized
    def get_optimizer(self):
        return self._get_optimizer()

class UNet_Model(ModelDesc):
    # # replace BatchNorm by LayerNorm
    # @auto_reuse_variable_scope
    # def discriminator(self, imgs):
    #     nf = 64
    #     with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
    #             argscope(LeakyReLU, alpha=0.2):
    #         l = (LinearWrap(imgs)
    #              .Conv2D('conv0', nf, nl=LeakyReLU)
    #              .Conv2D('conv1', nf * 2)
    #              .LayerNorm('ln1').LeakyReLU()
    #              .Conv2D('conv2', nf * 4)
    #              .LayerNorm('ln2').LeakyReLU()
    #              .Conv2D('conv3', nf * 8)
    #              .LayerNorm('ln3').LeakyReLU()
    #              .FullyConnected('fct', 1, nl=tf.identity)())
    #     return tf.reshape(l, [-1])

    def _get_inputs(self):
        #return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]
                #InputDesc(tf.int32, (None, opt.SHAPE, opt.SHAPE, 3), 'label')]
        return [InputDesc(tf.float32, (None, None, None, 1), 'input'),
         InputDesc(tf.float32, (None, None, None, 12), 'label')]


    def collect_variables(self, scope='reconstruction'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assert self.vars



    def reconstruction(self, imgs):
        NF = 64
        with argscope(Conv2D, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
                      #nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
            # encoder
            e1 = Conv2D('conv1', imgs, NF, nl=LeakyReLU)
            e2 = Conv2D('conv2', e1, NF * 2)
            e3 = Conv2D('conv3', e2, NF * 4)
            e4 = Conv2D('conv4', e3, NF * 8)
            e5 = Conv2D('conv5', e4, NF * 8)
            e6 = Conv2D('conv6', e5, NF * 8)
            e7 = Conv2D('conv7', e6, NF * 8)
            #e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            # decoder
            #e8 = Deconv2D('deconv1', e8, NF * 8)
            #e8 = Dropout(e8)
            #e8 = ConcatWith(e8, 3, e7)

            e7 = Deconv2D('deconv2', e7, NF * 8)
            e7 = Dropout(e7)
            e7 = ConcatWith(e7, e6,3)

            e6 = Deconv2D('deconv3', e7, NF * 8)
            e6 = Dropout(e6)
            e6 = ConcatWith(e6,  e5,3)

            e5 = Deconv2D('deconv4', e6, NF * 8)
            e5 = Dropout(e5)
            e5 = ConcatWith(e5,  e4,3)

            e4 = Deconv2D('deconv5', e5, NF * 4)
            e4 = Dropout(e4)
            e4 = ConcatWith(e4, e3,3)

            e3 = Deconv2D('deconv6', e4, NF * 2)
            e3 = Dropout(e3)
            e3 = ConcatWith(e3,  e2,3)

            e2 = Deconv2D('deconv7', e3, NF * 1)
            e2 = Dropout(e2)
            e2 = ConcatWith(e2, e1,3)

            prediction = Deconv2D('prediction', e2, 12, nl=tf.nn.relu)
        return prediction

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1
        target = inputs[1]/256.0
        #image_label = inputs[1]
        #image_label = image_label / 128.0 - 1

        #z = tf.random_normal([opt.BATCH, opt.Z_DIM], name='z_train')
        #z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('reconstruction'):
                prediction = self.reconstruction2(image_pos)
            #tf.summary.image('generated-samples', image_gen, max_outputs=30)

            #with tf.variable_scope('reconstruction'):
            #    prediction = self.generator(image_pos)

        self.cost = tf.nn.l2_loss(prediction - target, name="L2loss")
        # the Wasserstein-GAN losses
        # self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        # self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        #
        # # the gradient penalty loss
        # gradients = tf.gradients(vec_interp, [interp])[0]
        # gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        # gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        # gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        # add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)
        #
        # self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)
        add_moving_summary(self.cost)
        tf.summary.image('original', image_pos, max_outputs=30)
        tf.summary.image('prediction', prediction[:,:,:,0:3], max_outputs=30)
        tf.summary.image('target', target[:,:,:,0:3], max_outputs=30)

        self.build_losses()
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt

    def build_losses(self):
        """D and G play two-player minimax game with value function V(G,D)

          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
        """
        with tf.name_scope("L2_loss"):
            self.loss = self.cost
            #add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)

    #@memoized
    def get_optimizer(self):
        return self._get_optimizer()


class AutoEncoderTrainer(Trainer):
    def __init__(self, config):
        """
        GANTrainer expects a ModelDesc in config which sets the following attribute
        after :meth:`_build_graph`: g_loss, d_loss, g_vars, d_vars.
        """
        input = QueueInput(config.dataflow)
        model = config.model

        cbs = input.setup(model.get_inputs_desc())
        config.callbacks.extend(cbs)

        with TowerContext('', is_training=True):
            model.build_graph(input)
        opt = model.get_optimizer()

        # by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            rec_min = opt.minimize(model.loss, var_list=model.vars, name='g_op')
        self.train_op = rec_min

        super(AutoEncoderTrainer, self).__init__(config)

def get_augmentors():
    augs = []
    #if opt.load_size:
    #    augs.append(imgaug.Resize(opt.load_size))
    #if opt.crop_size:
    #    augs.append(imgaug.CenterCrop(opt.crop_size))
    #augs.append(imgaug.GaussianNoise(100))
    #augs.append(imgaug.RandomCrop(opt.SHAPE))
    augs.append(imgaug.Resize(opt.SHAPE))
    return augs

def get_data(datadir):
    imgs = glob.glob(datadir + '/*.npy') # outfile = data_path + subjectID + '_latent.npy'
    ds = ImageFromFile_AutoEcoder(imgs, channel=1, shuffle=True)
    #augmentor = get_augmentors()

    #ds = MultiThreadMapData(
    #    ds, nr_thread= 5,
    #    map_func=lambda dp: [augmentor.augment(dp[0]), dp[1]], buffer_size=1000)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 2)
    #ds = PrintData(ds, num=2)  # only for debugging
    return ds

def minutiae_extraction(model, model_path,sample_path, imgs, output_name='reconstruction/gen:0'):
    config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['sub:0'],#['z'],#
        #input_data_mapping=[0],
        output_names=[output_name, 'sub:0']) #,'reconstruction/feature:0'
    #imgs = glob.glob('/future/Data/Rolled/NSITSD14/Image2_jpeg' + '/*.jpeg')
    imgs = glob.glob('/media/kaicao/Data/Data/Rolled/NISTSD4/Image_Aligned'+'/*.jpeg')
    ds = ImageFromFile_AutoEcoder_Prediction(imgs, channel=1, shuffle=False)
    ds = PrefetchDataZMQ(ds, 1)
    ds = BatchData(ds, 1, remainder=True)
    #ds = PrintData(ds, num=2)  # only for debugging
    pred = SimpleDatasetPredictor(config, ds)
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        cv2.imwrite('test.jpeg', (o[0][0] + 1) * 128)
        cv2.imwrite('test2.jpeg', (o[1][0] + 1) * 128)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', help='model for minutiae extraction.', type=str, default='Cao_Model')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--inference', action='store_true', help='extract minutiae on input images')
    parser.add_argument('--image_dir', help='a jpeg directory',
                        default='/home/kaicao/Dropbox/Research/AutomatedLatentRecognition/Data/minutiae_cylinder_uint8')
    parser.add_argument('--sample_dir', help='a jpeg directory',
                        default='/home/kaicao/Dropbox/Research/AutomatedLatentRecognition/Data/minutiae_cylinder_uint8')
    parser.add_argument('--data', help='a jpeg directory',default='/home/kaicao/Dropbox/Research/AutomatedLatentRecognition/Data/minutiae_cylinder_uint8')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument('--log_dir', help='directory to save checkout point', type=str,
                        default='/home/kaicao/Research/AutomatedLatentRecognition/log_AutoEncoder/Minutiae_Cao')
    args = parser.parse_args()
    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.batch_size:
        opt.BATCH = args.batch_size
    return args

def get_config(log_dir,datadir,model):
    #logger.auto_set_dir()
    logger.set_logger_dir(log_dir)
    dataset = get_data(datadir)
    #lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
    if model is 'Cao_Model':
        return TrainConfig(
            dataflow=dataset,
            #optimizer=tf.train.AdamOptimizer(lr),
            #callbacks=[PeriodicTrigger(ModelSaver(), every_k_epochs=3)],
            callbacks=[ModelSaver(keep_recent=True)],
            model=Cao_Model(),
            steps_per_epoch=1000, #dataset.size()
            max_epoch=3000,
            session_init=SaverRestore(args.load) if args.load else None
        )
    else:
        return TrainConfig(
            dataflow=dataset,
            # optimizer=tf.train.AdamOptimizer(lr),
            # callbacks=[PeriodicTrigger(ModelSaver(), every_k_epochs=3)],
            callbacks=[ModelSaver(keep_recent=True)],
            model=UNet_Model(),
            steps_per_epoch=1000,  # dataset.size()
            max_epoch=3000,
            session_init=SaverRestore(args.load) if args.load else None
        )

if __name__ == '__main__':
    args = get_args()

    print(args)

    # imgs = glob.glob(args.data + '/*.npy')
    # matrix = np.load(imgs[0])
    # im = matrix[:, :, 0]  # np.squeeze(matrix[:,:,0])
    # cylinder = matrix[:, :, 2::]
    if args.inference and args.load:
        # model = get_model_loader(args.load)
        imgs = ['/future/Data/Rolled/selected_rolled_prints/MI0479144T_07/low_02_A103585608W_07.bmp']
        minutiae_extraction(AEC_Model(), args.load, args.sample_dir, imgs)
    else:
        config = get_config(args.log_dir,args.data,args.model)
        AutoEncoderTrainer(config).train()
#--inference --load /home/kaicao/Research/AutomatedLatentRecognition/log_AutoEncoder/UNet_minutiae/model-138000.index