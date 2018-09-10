#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import time
import os
import logging
import sys

from model import Model
from solver import Solver
import config


def process_args(args, defaults):
  def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser()

  parser.add_argument('--phase', dest='phase', type=str, default=defaults.PHASE, choices=['train', 'test'])
  parser.add_argument('--batch-size', dest='batch_size', type=int, default=defaults.BATCH_SIZE)
  parser.add_argument('--max-epoch', dest='max_epoch', type=int, default=defaults.MAX_EPOCH)
  parser.add_argument('--initial-learning-rate', dest="initial_lr", type=float,
                      default=defaults.INITIAL_LEARNING_RATE)
  parser.add_argument('--reg', dest='reg', type=float, default=defaults.REG,
                      help='regularization coefficient.')
  parser.add_argument('--dropout', dest='dropout', type=float, default=defaults.DROPOUT,
                      help='The probability that each element is kept.')
  parser.add_argument('--data-dir', dest='data_dir', type=str, default=defaults.DATA_DIR)
  parser.add_argument('--out-dir', dest='out_dir', type=str, default=defaults.OUT_DIR)
  parser.add_argument('--optimizer', dest='optimizer', type=str, default=defaults.OPTIMIZER)
  parser.add_argument('--restore-model-dir', dest='restore_model_dir', type=str,
                      default=defaults.RESTORE_MODEL_DIR)

  parser.add_argument('--neg-ratio', dest='neg_ratio', type=float, default=defaults.NEG_RATIO)
  parser.add_argument('--item-dim', dest='item_dim', type=int, default=defaults.ITEM_DIM)
  parser.add_argument('--cate-dim', dest='cate_dim', type=int, default=defaults.CATE_DIM)
  parser.add_argument('--user-dim', dest='user_dim', type=int, default=defaults.USER_DIM)
  parser.add_argument('--max-gradient-norm', dest='max_gradient_norm', type=float, default=5.0)
  parser.add_argument('--display', dest='display', type=int, default=defaults.DISPLAY)
  parser.add_argument('--n-block', dest='n_block', type=int, default=defaults.N_BLOCK)
  parser.add_argument('--fusion-layers', dest='fusion_layers', type=int, nargs='+',
                      default=defaults.FUSION_LAYERS)
  parser.add_argument('--num-heads', dest='num_heads', type=int, default=defaults.NUM_HEADS)
  parser.add_argument('--max-length', dest='max_length', type=int, default=defaults.MAX_LENGTH)

  parameters = parser.parse_args(args)
  return parameters

def init_logging(params):
  log_name = os.path.join(params.out_dir, '{}.log'.format(time.strftime('%Y%m%d-%H%M')))

  logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename=log_name,
  )
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

  logging.info('phase: {}'.format(params.phase))
  logging.info('batch_size: {}'.format(params.batch_size))
  logging.info('max_epoch: {}'.format(params.max_epoch))
  logging.info('initial_learning_rate: {}'.format(params.initial_lr))
  logging.info('reg: {:.8f}'.format(params.reg))
  logging.info('optimizer: {}'.format(params.optimizer))
  logging.info('dropout: {:.2f}'.format(params.dropout))
  logging.info('data_dir: {}'.format(params.data_dir))
  logging.info('out_dir: {}'.format(params.out_dir))
  logging.info('optimizer: {}'.format(params.optimizer))
  logging.info('restore_model_dir: {}'.format(params.restore_model_dir))
  logging.info('neg_ratio: {}'.format(params.neg_ratio))
  logging.info('max_length: {}'.format(params.max_length))
  logging.info('user_dim: {}'.format(params.user_dim))
  logging.info('item_dim: {}'.format(params.item_dim))
  logging.info('cate_dim: {}'.format(params.cate_dim))
  logging.info('num_heads: {}'.format(params.num_heads))
  logging.info('fusion_layers: {}'.format('|'.join([str(i) for i in params.fusion_layers])))
  logging.info('--------------------')


def main(args, defaults):
  param = process_args(args, defaults)
  directory = '{}'.format("THACIL")
  param.out_dir = os.path.join(param.out_dir, directory)
  if not os.path.exists(param.out_dir):
    os.makedirs(param.out_dir)

  init_logging(param)
  model = Model(param)
  solver = Solver(model, param)
  if param.phase == 'train':
    solver.train()
  elif param.phase == 'test':
    solver.test()
  else:
    raise ValueError("'phase' should be 'train' or 'eval'")

if __name__ == '__main__':
    main(sys.argv[1:], config.Config)



























