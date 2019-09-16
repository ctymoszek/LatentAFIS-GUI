#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Improved-WGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.globvars import globalns as G
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from tensorpack.utils.globvars import globalns as opt

from GAN import SeparateGANTrainer

"""
Improved Wasserstein-GAN.
See the docstring in DCGAN.py for usage.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN, and change the batch size & model
import DCGAN
G.BATCH = 32 
G.Z_DIM = 512 


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

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
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
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random_normal([G.BATCH, G.Z_DIM], name='z_train')
        z = tf.placeholder_with_default(z, [None, G.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)

            alpha = tf.random_uniform(shape=[G.BATCH, 1, 1, 1],
                                      minval=0., maxval=1., name='alpha')
            interp = image_pos + alpha * (image_gen - image_pos)

            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)
                vec_interp = self.discriminator(interp)

        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')

        # the gradient penalty loss
        gradients = tf.gradients(vec_interp, [interp])[0]
        gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)

        self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)

        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model',default='model/I-WGAN_CAE/model-620000.index')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--sample_dir', help='directory for generated examples',type=str,default='/media/kaicao/Data/Data/FingerprintSynthesis/tensorpack/I-WGAN_CAE_10M_JPEG/')
    parser.add_argument('--data', help='a jpeg directory',default='/media/kaicao/Data/Data/Rolled/MSP/Image_Aligned/')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument('--log_dir', help='directory to save checkout point', type=str,
                        default='/media/kaicao/Data/checkpoint/FingerprintSynthesis/tensorpack/Improved-WGAN_AutoEncoder/')
    args = parser.parse_args()
    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()
    if args.sample:
        DCGAN.sample2(Model(), args.load,args.sample_dir,num=10000000)
    else:
        assert args.data
        #logger.auto_set_dir()
        logger.set_logger_dir(args.log_dir)
        #a = SaverRestore(args.load)
        config = TrainConfig(
            model=Model(),
            dataflow=DCGAN.get_data(args.data),
            callbacks=[ModelSaver()],
            steps_per_epoch=1000,
            max_epoch=500,
            session_init=SaverRestore(args.load) if args.load else None
        )
        SeparateGANTrainer(config, g_period=6).train()
