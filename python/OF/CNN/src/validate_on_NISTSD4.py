"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import glob

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            # Get the paths for the corresponding images
            #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.test_dir), pairs, args.lfw_file_ext)
            paths = glob.glob(os.path.expanduser(args.test_dir)+'*.jpeg')
            paths.sort()
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("batch_join:0")
            #embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #images_placeholder = tf.get_default_graph4().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("Add:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #prelogs = tf.get_default_graph().get_tensor_by_name("Add_1:0")
            #class_num = prelogs.get_shape()[1]
            #print('no. of classes: %d' %class_num)
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on NISTSD4 images')
            batch_size = args.batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                print(i)
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size,do_prewhiten=False)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            for i in range(nrof_images):
                norm = np.linalg.norm(emb_array[i, :]) +0.0000001
                emb_array[i, :] = emb_array[i, :]/norm
            nrof_subjects = (int)(nrof_images//2)
            embeddings1 = emb_array[:nrof_subjects]
            embeddings2 = emb_array[nrof_subjects:nrof_subjects*2]
            rank = np.zeros((nrof_subjects,))
            for i in range(nrof_subjects):
                diff = np.subtract(np.tile(embeddings1[i, :], (nrof_subjects, 1)), embeddings2)
                dist = np.sum(np.square(diff), 1)
                ind = np.argsort(dist)
                rank[i] = np.where(ind == i)[0] + 1
            print(len(rank[rank == 1]) * 1.0 / nrof_subjects)
            print(len(rank[rank <= 10]) * 1.0 / nrof_subjects)
            #tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
            #    actual_issame, nrof_folds=args.lfw_nrof_folds)

            #print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            #print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            #auc = metrics.auc(fpr, tpr)
            #print('Area Under Curve (AUC): %1.3f' % auc)
            #eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            #print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',
                        default='/media/kaicao/Data/Data/Rolled/NISTSD4/Image_Aligned_2_0.65/')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/media/kaicao/Data/models/indexing/20170708-220143')#''../models/triplet_debug/20170709-000103/')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
