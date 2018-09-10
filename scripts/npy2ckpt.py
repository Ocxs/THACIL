#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# __date__ = 18-8-1:16-22
# __author__ = Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import numpy as np
import os
import argparse



def npy2ckpt(npy_path, ckpt_path):
  from tensorflow.core.protobuf import saver_pb2
  import tensorflow as tf

  feature = np.load(npy_path)
  W = tf.Variable(
    tf.constant(0.0, shape=feature.shape),
    trainable=False,
    name='train_cover_image_feature'
  )
  emb_ph = tf.placeholder(tf.float32, feature.shape)
  emb_init = W.assign(emb_ph)
  tf.add_to_collection('param', W)

  with tf.Session() as sess:
    saver = tf.train.Saver(tf.get_collection('param'), write_version=saver_pb2.SaverDef.V1)
    sess.run(tf.global_variables_initializer())
    sess.run(emb_init, feed_dict={emb_ph: feature})
    saver.save(sess, os.path.join(ckpt_path, 'train_cover_image_feature.ckpt'))


def main(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('--npy-path', dest='npy_path', type=str, help='path of npy file')
  parser.add_argument('--ckpt-path', dest='ckpt_path', type=str, help='path where to save ckpt file')
  params = parser.parse_args(args)

  npy2ckpt(params.npy_path, params.ckpt_path)


if __name__ == '__main__':
  main(sys.argv[1:])
