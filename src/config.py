#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config:
  PHASE = 'test'

  BATCH_SIZE = 128
  MAX_EPOCH = 20
  INITIAL_LEARNING_RATE = 0.001
  REG = 0.00005
  DROPOUT = 0.8

  DATA_DIR = '../data/input'
  OUT_DIR = '../data/output'
  RESTORE_MODEL_DIR = '../data/input'
  OPTIMIZER = 'adam'

  NEG_RATIO = 1.0
  FUSION_LAYERS = [128]

  MAX_LENGTH = 160
  ITEM_DIM = 64
  CATE_DIM = 64
  USER_DIM = 128
  DISPLAY = 10
  NUM_HEADS = 8
  N_BLOCK = 8







