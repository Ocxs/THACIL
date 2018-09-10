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
from tqdm import tqdm


def read_data(path):
  data = []
  with open(path, 'r') as reader:
    reader.readline()
    for line in tqdm(reader):
      lines = line.strip('\n').split(',')
      items = [int(i) for i in lines]
      data.append(items)
  return data

def generate_user_click_ids_npy(train_data_path, save_path):
  train_data = read_data(train_data_path)
  user_click_ids = [[] for _ in range(10986)]
  for item in tqdm(train_data):
    if item[3] == 1:
      user_click_ids[item[0]].append((item[1], item[2], item[-1]))

  sorted_user_click_ids = [sorted(item, key=lambda x: x[-1]) for item in user_click_ids]
  np.save(os.path.join(save_path, 'user_click_ids.npy'), sorted_user_click_ids)


def main(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-data-path', dest='train_data_path', type=str, help='path of train_data.csv')
  parser.add_argument('--save-path', dest='save_path', type=str, help='path where to save user_click_ids.npy')
  params = parser.parse_args(args)

  generate_user_click_ids_npy(params.train_data_path, params.save_path)

if __name__ == '__main__':
  main(sys.argv[1:])
