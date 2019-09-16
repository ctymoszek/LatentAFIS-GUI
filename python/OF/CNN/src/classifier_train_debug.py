"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw

slim = tf.contrib.slim

def _fully_connected(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    matrix = tf.get_variable(name+"Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(name+"Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    output_vector = tf.add(tf.matmul(input_, matrix), bias)
    return output_vector

def main(args):

    #network = importlib.import_module(args.model_def, 'inception_v3')
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)


    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg + ' ' + str(getattr(args, arg)) + '\n')

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)

        # Read data and apply label preserving distortions
        image_batch, label_batch = facenet.read_and_augument_data(image_list, label_list, args.image_size,
            args.batch_size, args.max_nrof_epochs, args.random_rotate,args.random_crop, args.random_flip, args.nrof_preprocess_threads)
        print('Total number of classes: %d' % len(train_set))
        print('Total number of examples: %d' % len(image_list))

        # Node for input images
        image_batch = tf.identity(image_batch, name='input')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Placeholder for keep probability
        keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_prob')

        # Build the inference graph
        # prelogits = network.inference(image_batch, keep_probability_placeholder,
        #    phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
        batch_norm_params = {
            # Decay for the moving averages
            'decay': 0.995,
            # epsilon to prevent 0s in variance
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
            # Only update statistics during training mode
            'is_training': phase_train_placeholder
        }

        #prelogits, _ = network.inception_v3(image_batch, num_classes=len(train_set),is_training=True)
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
        #prelogits = tf.identity(prelogits, name="prelogits")
        bottleneck =  _fully_connected(prelogits, args.embedding_size,name='embedding')
        logits = _fully_connected(bottleneck, len(train_set),name='logits')
        """
        bottleneck = slim.fully_connected(prelogits, args.embedding_size, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                scope='Bottleneck', reuse=False)

        logits = slim.fully_connected(bottleneck, len(train_set), activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

        logits = tf.identity(logits, name="logits")
        """
        # Add DeCov regularization loss
        if args.decov_loss_factor>0.0:
            logits_decov_loss = facenet.decov_loss(logits) * args.decov_loss_factor
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, logits_decov_loss)

        # Add center loss
        update_centers = tf.no_op('update_centers')
        if args.center_loss_factor>0.0:
            prelogits_center_loss, update_centers = facenet.center_loss(bottleneck, label_batch, args.center_loss_alfa,
                nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        #embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        """
        # Multi-label loss: sigmoid loss
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=logits, name='sigmoid_loss_per_example')
        sigmoid_loss_mean = tf.reduce_mean(sigmoid_loss, name='sigmoid_loss')
        """
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        #total_loss = tf.add_n([cross_entropy_mean], name='total_loss')

        # prediction
        prediction = tf.argmax(logits, axis=1, name='prediction')
        acc = slim.metrics.accuracy(predictions=tf.cast(prediction, dtype=tf.int32),
                                    labels=tf.cast(label_batch, dtype=tf.int32))


        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        # save_variables = list(set(tf.all_variables())-set([w])-set([b]))
        save_variables = tf.trainable_variables()
        saver = tf.train.Saver(save_variables, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.train.start_queue_runners(sess=sess)

        with sess.as_default():

            if pretrained_model:
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, phase_train_placeholder, learning_rate_placeholder, keep_probability_placeholder, global_step,
                    total_loss, acc,train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                    update_centers)

                # Evaluate on LFW
                if args.lfw_dir:
                    start_time = time.time()
                    _, _, accuracy, val, val_std, far = lfw.validate(sess, lfw_paths, actual_issame, args.seed,
                        args.batch_size, image_batch, phase_train_placeholder, keep_probability_placeholder, embeddings, nrof_folds=args.lfw_nrof_folds)
                    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                    lfw_time = time.time() - start_time
                    # Add validation loss and accuracy to summary
                    summary = tf.Summary()
                    #pylint: disable=maybe-no-member
                    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
                    summary.value.add(tag='lfw/val_rate', simple_value=val)
                    summary.value.add(tag='time/lfw', simple_value=lfw_time)
                    summary_writer.add_summary(summary, step)
                    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
                        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir

def train(args, sess, epoch, phase_train_placeholder, learning_rate_placeholder, keep_probability_placeholder, global_step,
      loss, acc, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, update_centers):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Perform training on the selected triplets
        train_time = 0
        i = 0
        while batch_number < args.epoch_size:
            start_time = time.time()
            feed_dict = {phase_train_placeholder: True, learning_rate_placeholder: lr, keep_probability_placeholder: args.keep_probability}
            err, _, _, step, reg_loss, rank_1 = sess.run([loss, train_op, update_centers, global_step, regularization_losses,acc], feed_dict=feed_dict)
            if (batch_number % 100 == 0):
                summary_str, step = sess.run([summary_op, global_step], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\tRank-1 %2.3f \t LearningRate %2.3f'  %
                  (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss),rank_1,lr))
            batch_number += 1
            i += 1
            train_time += duration
        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/total', simple_value=train_time)
        summary_writer.add_summary(summary, step)
    return step

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_size', type=int,
        help='embeddihg size.', default=512)
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='../logs/indexing_debug')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='../models/indexing_debug')
    parser.add_argument('--sub_dir', type=str,
        help='Directory where to put real data', default='resnet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    #,default ='../models/indexing_debug/20170629-112608/model-20170629-112608.ckpt-229000.data-00000-of-00001'
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/Research/Data/Rolled/Longitudinal/aligned_image_rearranged_0.65/')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=1000)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=299)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop',
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true',default=True)
    parser.add_argument('--random_flip',
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
        help='Performs random rotation of training images.', action='store_true',default=True)
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=1e-5)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.00)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.6)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=-1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='../data/learning_rate_schedule_classifier_casia.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
