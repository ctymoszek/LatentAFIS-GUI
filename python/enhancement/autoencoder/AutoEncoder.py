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

#from GAN import GANTrainer, RandomZData, GANModelDesc
import DCGAN
import scipy.misc
from tensorpack import Trainer
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

from tensorpack import *
from tensorpack.utils.globvars import globalns as opt
import glob, os



class Model(DCGAN.Model):
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

    def collect_variables(self, g_scope='feature', d_scope='gen'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        assert self.g_vars
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)
        assert self.d_vars

        self.vars = self.g_vars +self.d_vars

    @auto_reuse_variable_scope
    def extraction(self, imgs):
        """ return a (b, 1) logits"""
        nf = 16
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
             l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .LayerNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .LayerNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .LayerNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', nf * 16)
                 .LayerNorm('bn4').LeakyReLU()
                 .Conv2D('conv5', nf * 32)
                 .LayerNorm('bn5').LeakyReLU()
                 .Conv2D('conv6', nf * 64)
                 .LayerNorm('bn6').LeakyReLU()
                 .FullyConnected('fct', opt.Z_DIM, nl=tf.identity)())
             l = tf.tanh(l, name='representation')
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1


        #z = tf.random_normal([opt.BATCH, opt.Z_DIM], name='z_train')
        #z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('feature'):
                feature = self.extraction(image_pos)
            #tf.summary.image('generated-samples', image_gen, max_outputs=30)

            with tf.variable_scope('gen'):
                prediction = self.generator(feature)

        self.cost = tf.nn.l2_loss(prediction - image_pos, name="L2loss")
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
        tf.summary.image('prediction', prediction, max_outputs=30)
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
    augs.append(imgaug.Resize(opt.SHAPE))
    return augs

def get_data(datadir):
    imgs = glob.glob(datadir + '/*.jpeg')
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 5)
    #ds = PrintData(ds, num=2)  # only for debugging
    return ds



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory',default='/media/kaicao/Data/Data/Rolled/MSP/Image_Aligned')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument('--log_dir', help='directory to save checkout point', type=str,
                        default='/media/kaicao/Data/checkpoint/FingerprintSynthesis/tensorpack/AutoEncoder/')
    args = parser.parse_args()
    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

def get_config(log_dir,datadir):
    #logger.auto_set_dir()
    logger.set_logger_dir(log_dir)
    dataset = get_data(datadir)
    #lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
    return TrainConfig(
        dataflow=dataset,
        #optimizer=tf.train.AdamOptimizer(lr),
        #callbacks=[PeriodicTrigger(ModelSaver(), every_k_epochs=3)],
        callbacks=[ModelSaver()],
        model=Model(),
        steps_per_epoch=1000, #dataset.size()
        max_epoch=3000,
        session_init=SaverRestore(args.load) if args.load else None
    )

if __name__ == '__main__':
    args = get_args()
    print(args)
    config = get_config(args.log_dir,args.data)
    AutoEncoderTrainer(config).train()