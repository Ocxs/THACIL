#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

def var_init(name, shape, initializer=tf.contrib.layers.xavier_initializer(),
              trainable=True):
  with tf.device('/gpu:0'):
    var = tf.get_variable(
      name=name,
      shape=shape,
      initializer=initializer,
      trainable=trainable
    )
    if not tf.get_variable_scope().reuse and name != 'train_cover_image_feature':
      tf.add_to_collection("parameters", var)
    if name == 'train_cover_image_feature':
      tf.add_to_collection('train_cover_image_feature', var)
    return var

def dense(x,
          units,
          name,
          keep_prob=1.0,
          activation= None,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          bias_initializer=tf.zeros_initializer(),
          reuse=None):
  """
  Functional interface for the densely-connected layer.
  :param x: Tensor input.
  :param units: Integer or Long, dimensionality of the output space.
  :param name: String, the name of the parameter.
  :param keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
  :param activation: Activation function (callable). Set it to None to maintain a linear activation.
  :param kernel_initializer:
  :param bias_initializer:
  :param reuse:
  :return:
  """
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    if not isinstance(name, (list, tuple)):
      raise ValueError('name should be list or tuple')

    prev_units= x.get_shape().as_list()[-1]
    w = var_init(name[0], (prev_units, units), kernel_initializer)
    out = tf.tensordot(x, w, axes=[[-1], [0]])
    if len(name) > 1:
      b = var_init(name[1], units, bias_initializer)
      out += b

    if activation is not None:
      out = activation(out)

    out = tf.nn.dropout(out, keep_prob)
    return out


def temporal_hierarchical_attention(att_cate_emb,
                                    att_item_emb,
                                    intra_mask,
                                    inter_mask,
                                    num_heads,
                                    keep_prob):
  """
  Perform temporal hierarchical attention on user clicked videos.
  :param att_cate_emb: The categorical feature of user clicked videos, [N, max_length, cate_dim]
  :param att_item_emb: The visual feature of cover image of user clicked videos, [N, max_length, item_dim]
  :param intra_mask: The mask of user click behavior, [N, max_length], handle the case which the number of
                     user clicked video less than max_length.
  :param inter_mask: The mask of user click behavior, [N, n_block], handle the case which the
                     number of user clicked video less than (max_length - max_length/n_block).
  :param num_heads: The number of heads in multi-head attention.
  :param keep_prob:
  :return: User profiles.
  """

  n_block = inter_mask.get_shape().as_list()[1]

  item_block_emb = tf.concat(tf.split(tf.expand_dims(att_item_emb,axis=1), n_block, axis=2), axis=1)
  cate_block_emb = tf.concat(tf.split(tf.expand_dims(att_cate_emb,axis=1), n_block, axis=2), axis=1)
  intra_block_mask = tf.concat(tf.split(tf.expand_dims(intra_mask,axis=1), n_block, axis=2), axis=1)

  with tf.variable_scope('intra_attention'):
    with tf.variable_scope('item_attention'):
      item_att_emb = additive_vanilla_attention(item_block_emb,
                                                cate_block_emb,
                                                intra_block_mask,
                                                keep_prob)
    with tf.variable_scope('cate_attention'):
      cate_att_emb = additive_vanilla_attention(cate_block_emb,
                                                item_block_emb,
                                                intra_block_mask,
                                                keep_prob)  # (N, cate_dim)
    # local information
    intra_vec = tf.concat([item_att_emb, cate_att_emb], axis=-1)  # (N, n_block, item_dim+cate_dim)

  with tf.variable_scope('inter_attention'):
    # global information
    inter_vec = forward_multi_head_self_attention(intra_vec, inter_mask, keep_prob, num_heads)

  with tf.variable_scope('feature_fusion'):
    output = intra_vec + inter_vec
    user_profile = normalize(output)
  return user_profile

def forward_multi_head_self_attention(x, mask, keep_prob, num_heads=8):
  """
  Perform forward multi-head self-attention to generate global information.
  :param x: Local information l within each block.  [N, n_block, dim]
  :param mask: Inter_mask in temporal_hierarchical_attention function
  :param keep_prob:
  :param num_heads:
  :return: Global information g within each block.
  """
  batch_size, length, dim = x.get_shape().as_list()

  # linear projection
  q = dense(x, dim, ['w_q'], keep_prob)  # [n,m,d], m=n_block
  k = dense(x, dim, ['w_k'], keep_prob)  # [n,m,d]
  v = dense(x, dim, ['w_v'], keep_prob)  # [n,m,d]

  # multi-head
  q = tf.concat(tf.split(q, num_heads, axis=2), axis=0) # [n*h, m, d/h]
  k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)  # [n*h, m, d/h]

  # additive attention
  q = tf.expand_dims(q, axis=1)  # [n*h,1,m,d/h]
  k = tf.expand_dims(k, axis=2)  # [n*h,m,1,d/h]
  bias = var_init('b', dim/num_heads)
  w = scaled_tanh(q+k+bias, 5.0) # [n*h,m,m,d/h]
  w = dense(w, dim/num_heads, ['w_p'], keep_prob)  # [n*h, m, m, d/h]

  # For computational efficiency
  w =  tf.concat(tf.split(w, num_heads, axis=0), axis=3)  # [n, m, m, d]
  v = tf.tile(tf.expand_dims(v, 1), [1, length, 1, 1])  # [n, m, m, d]

  # generate forward mask
  indices = tf.range(length, dtype=tf.int32)
  col, row = tf.meshgrid(indices, indices)
  forward_mask = tf.greater(row, col)
  forward_mask = tf.tile(tf.expand_dims(forward_mask, axis=0), [batch_size, 1, 1])
  mask = tf.tile(tf.expand_dims(mask, axis=1), [1, length, 1])
  att_mask = tf.logical_and(forward_mask, mask)  # [n,m,m]

  mask_w = softmax_mask(w, att_mask)
  alpha = tf.nn.softmax(mask_w, dim=2)  # [n,m,m,d]
  alpha = mask01(alpha, mask)
  att_vec = tf.reduce_sum(alpha*v, axis=2)
  return att_vec

def scaled_tanh(x, scale=5.):
  return scale * tf.nn.tanh(1. / scale * x)

def additive_vanilla_attention(x, q, mask, keep_prob):
  x_shape = x.get_shape().as_list()
  q_shape = q.get_shape().as_list()
  x_proj = dense(x, x_shape[-1], ['x_w1', 'x_b1'], keep_prob)
  q_proj = dense(q, x_shape[-1], ['q_w1', 'q_b1'], keep_prob)
  if len(x_shape) - len(q_shape) == 2:
    q_proj = tf.expand_dims(tf.expand_dims(q_proj, -2), -2)
  elif len(x_shape) - len(q_shape) == 1:
    q_proj = tf.expand_dims(q_proj, -2)
  proj = tf.nn.relu(x_proj + q_proj)
  w = dense(proj, x_shape[-1], ['w2', 'b2'], keep_prob)
  mask_w = softmax_mask(w, mask)
  alpha = tf.nn.softmax(mask_w, dim=len(x_shape)-2)
  att_vec = tf.reduce_sum(alpha * x, axis=len(x_shape)-2)
  return att_vec

def softmax_mask(x, mask):
  """
  :param x: [n,m,d] or [n,b,m,d]
  :param mask: [n,m] or [n,b,m]
  :return:
  """
  x_shape = x.get_shape().as_list()
  pad_num = len(x_shape)-1
  mask = tf.tile(tf.expand_dims(mask, axis=-1), pad_num*[1]+[x_shape[-1]])
  paddings = tf.ones_like(mask, tf.float32) * (-2 ** 32 + 1)
  softmax_mask = tf.where(mask, x, paddings)
  return softmax_mask

def mask01(x, mask):
  """
  :param x: [n,m,d] or [n,b,m,d]
  :param mask: [n,m] or [n,b,m]
  :return:
  """
  x_shape = x.get_shape().as_list()
  pad_num = len(x_shape)-1
  mask = tf.tile(tf.expand_dims(mask, axis=-1), pad_num*[1]+[x_shape[-1]])
  mask_x = tf.where(mask, x, tf.cast(mask, tf.float32))
  return mask_x

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
  '''Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
    `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable('beta', shape=params_shape)
    gamma = tf.get_variable('gamma', shape=params_shape)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    outputs = gamma * normalized + beta

  return outputs

def vanilla_attention(user_profile,
                      item_emb,
                      inter_mask,
                      keep_prob):
  """
  Perform vanilla attention to fuse user profiles
  :param user_profile: Generated from our temporal hierarchical attention.
  :param item_emb: The embedding of candidate item.
  :param inter_mask: Inter_mask is same as inter_mask in temporal_hierarchical_attention function.
  :param keep_prob:
  :return: User representation.
  """
  with tf.variable_scope('vanilla_attention'):
    user_emb = additive_vanilla_attention(user_profile, item_emb, inter_mask, keep_prob)
    return user_emb

def dnn(x,
        fusion_layers,
        keep_prob):
  """
  Feedforward network.
  :param x: Tensor input.
  :param fusion_layers: List, the layers of feedforward network, like [256, 128].
  :param keep_prob:
  :return: The micro-video click-through probabilities.
  """
  _, dim = x.get_shape().as_list()
  n_layers = len(fusion_layers)
  for i in range(n_layers):
    x = dense(x, fusion_layers[i], ['w{}'.format(i+1), 'b{}'.format(i+1)], keep_prob, tf.nn.relu)

  logit = dense(x, 1, ['w{}'.format(n_layers+1), 'b{}'.format(n_layers+1)], keep_prob)
  return tf.squeeze(logit)


