#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# __date__ = 17-10-24:20-08
# __author__ = Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import random, logging, os

class DataLoader(object):
  def __init__(self, params):
    self.data_dir = params.data_dir
    self.batch_size = params.batch_size
    self.n_block = params.n_block
    self.max_length = params.max_length

    self.block_size = self.max_length//self.n_block

    train_data_path = os.path.join(params.data_dir, 'train_data.csv')
    test_csv_path = os.path.join(params.data_dir, 'test_data.csv')

    self.preload_feat_into_memory()
    if params.phase =='train':
      self.read_train_data(train_data_path)
    self.read_test_data(test_csv_path)


  def get_train_batch(self, idx):
    batch_data = self.epoch_train_data[idx*self.batch_size: (idx+1)*self.batch_size]
    user_ids, item_ids, cate_ids, labels = zip(*batch_data)
    att_iids, att_cids, intra_mask, inter_mask = self.get_att_ids(user_ids)
    return user_ids, item_ids, cate_ids, att_iids, att_cids, intra_mask, inter_mask, labels


  def get_test_batch(self, idx):
    batch_data = self.test_data[idx*self.batch_size: (idx+1)*self.batch_size]
    user_ids, item_ids, cate_ids, labels = zip(*batch_data)
    item_vecs = self.get_test_cover_img_feature(item_ids)
    att_iids, att_cids, intra_mask, inter_mask = self.get_att_ids(user_ids)
    return user_ids, item_vecs, cate_ids, att_iids, att_cids, intra_mask, inter_mask, labels, item_ids


  def generate_train_data(self, neg_ratio=3):
    logging.info('generate samples for training')
    epoch_train_data = []
    for item in self.train_data:
      pos_num, neg_num = len(item[0]), len(item[1])
      epoch_train_data.extend(item[0])
      if neg_num < pos_num * neg_ratio:
        epoch_train_data.extend(item[1])
      else:
        epoch_train_data.extend(random.sample(item[1], pos_num*neg_ratio))

    random.shuffle(epoch_train_data)
    self.epoch_train_data = epoch_train_data
    self.epoch_train_length = len(epoch_train_data)


  def read_train_data(self, data_list_path):
    logging.info('start read data list from disk')
    with open(data_list_path, 'r') as reader:
      reader.readline()
      raw_data = map(lambda x: x.strip('\n').split(','), reader.readlines())


    data = [[[], []] for _ in range(10986)]
    for item in raw_data:
      if int(item[3]) == 1:
        data[int(item[0])][0].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
      else:
        data[int(item[0])][1].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    self.train_data = data


  def read_test_data(self, test_data_path, sep=','):
    with open(test_data_path, 'r') as reader:
      reader.readline()
      lines = map(lambda x: x.strip('\n').split(sep), reader.readlines())
      data = map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3])), lines)

    self.test_data = list(data)
    self.epoch_test_length = len(self.test_data)
    logging.info('{} test samples'.format(self.epoch_test_length))


  def sample_vid(self, tuples):
    item_ids, cate_ids, timestamps = list(zip(*tuples))
    length = len(item_ids)
    padding_num = self.max_length - length
    if padding_num > 0:
      item_ids = list(item_ids) + [984983] * padding_num
      cate_ids = list(cate_ids) + [512] * padding_num
      intra_mask =  [1] * length + [0] * padding_num
      pad_n_block = padding_num // self.block_size
      inter_mask = [1] * (self.n_block-pad_n_block) + [0] * pad_n_block
    else:
      indices = random.sample(list(range(length)), self.max_length)
      indices.sort()
      item_ids = [item_ids[i] for i in indices]
      cate_ids = [cate_ids[i] for i in indices]
      intra_mask = [1] * self.max_length
      inter_mask = [1] * self.n_block

    return item_ids, cate_ids, intra_mask, inter_mask


  def get_att_ids(self, user_ids):
    xx = [self.sample_vid(self.user_click_ids[uid]) for idx, uid in enumerate(user_ids)]
    batch_att_iids, batch_att_cids, batch_intra_mask, batch_inter_mask = zip(*xx)
    return batch_att_iids, batch_att_cids, batch_intra_mask, batch_inter_mask


  def get_test_cover_img_feature(self, vids):
    head_vec = [self.test_cover_img_feat[i] for i in vids]
    return head_vec


  def preload_feat_into_memory(self):
    test_feature_path = os.path.join(self.data_dir, 'test_cover_image_feature.npy')
    self.test_cover_img_feat = np.load(test_feature_path)
    logging.info('load test head feature')
    user_click_ids_path = os.path.join(self.data_dir, 'user_click_ids.npy')
    self.user_click_ids = np.load(user_click_ids_path)

