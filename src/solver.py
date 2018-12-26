#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import time

import tensorflow as tf
import numpy as np

from data_loader import DataLoader

class Solver(object):
  def __init__(self, model, params):
    self.model = model
    self.phase = params.phase
    self.max_epoch = params.max_epoch
    self.batch_size = params.batch_size
    self.display = params.display
    self.restore_model_dir = params.restore_model_dir
    self.initial_lr = params.initial_lr
    self.out_dir = params.out_dir
    self.neg_ratio = params.neg_ratio
    self.summary_dir = os.path.join(self.out_dir, 'summary')
    self.model_dir = os.path.join(self.out_dir, 'ckpt')


    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.summary_dir):
      os.makedirs(self.summary_dir)

    self.data= DataLoader(params)


  def create_model(self, sess):
    sess.run(tf.global_variables_initializer())
    params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
    saver = tf.train.Saver(params, max_to_keep=2)
    self.model.restore_train_visual_emb(self.data.train_visual_feature, sess)
    self.data.del_temp()

    if self.phase == 'test':
      has_ckpt = tf.train.get_checkpoint_state(self.model_dir)
      if has_ckpt:
        model_path = has_ckpt.model_checkpoint_path
        saver.restore(sess, model_path)
        logging.info("Load model from {}".format(model_path))
      else:
        raise ValueError("No checkpoint file found in {}".format(self.model_dir))
    else:
      logging.info('Create model with fresh parameters')

    now = time.strftime("%Y%m%d-%H%M")
    if self.phase == 'train':
      self.train_writer = tf.summary.FileWriter(
        os.path.join(self.summary_dir, '{}-train'.format(now)),
        sess.graph
      )
    self.test_writer = tf.summary.FileWriter(
      os.path.join(self.summary_dir, '{}-test'.format(now)),
      sess.graph
    )
    logging.info('create summary writer successfully')



  def train(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      self.create_model(sess)

      min_loss, stop_training_counter = 1e5, 0
      lr = self.initial_lr

      for epoch in range(self.max_epoch):
        self.data.generate_train_data(self.neg_ratio)
        epoch_train_length = self.data.epoch_train_length
        loop_num = int(epoch_train_length // self.batch_size)
        logging.info('start train phase')
        logging.info('train iterations: {}'.format(loop_num))
        avg_loss, avg_acc = 0.0, 0.0
        load_times, run_times = 0.0, 0.0
        epoch_avg_loss = 0.0
        for i in range(loop_num):
          start = time.time()
          batch_data = self.data.get_train_batch(i)
          load_time = time.time() - start
          loss, acc, summaries = self.model.train(sess, batch_data, lr)
          run_time = time.time() - start - load_time
          self.train_writer.add_summary(summaries, i + 1 + epoch * loop_num)

          load_times += load_time
          run_times += run_time
          avg_loss += loss
          avg_acc += acc
          epoch_avg_loss += loss
          if (i+1) % self.display == 0:
            logging.info('epoch {}-train step {}: loss: {:.3f}, acc: {:.3f} in {:.3f}s, load {:.3f}s'.format(
              epoch+1, i+1, avg_loss/self.display, avg_acc/self.display, run_times,load_times))
            load_times, run_times, avg_acc, avg_loss = 0.0, 0.0, 0.0, 0.0
            #break
        avg_loss, avg_acc = 0.0, 0.0
        load_times, run_times = 0.0, 0.0
        epoch_avg_loss /= loop_num
        logging.info('lr: {}, min loss: {}'.format(lr, min_loss))
        if epoch_avg_loss < min_loss:
          if min_loss - epoch_avg_loss < 0.005:
            stop_training_counter += 1
            lr *= 0.5
          else:
            stop_training_counter = max(0, stop_training_counter - 1)
          min_loss = epoch_avg_loss
          save_path = os.path.join(self.model_dir, 'model-{:.4f}.ckpt'.format(min_loss))
          self.model.save(sess, save_path, epoch + 1)
        else:
          stop_training_counter += 1
          lr *= 0.5


        if stop_training_counter > 5:
          logging.info('start test phase')
          epoch_test_length = self.data.epoch_test_length
          loop_num = int(epoch_test_length // self.batch_size)
          logging.info('test iterations: {}'.format(loop_num))
          pred_dict = {}
          for step in range(loop_num):
            start = time.time()
            batch_data = self.data.get_test_batch(step)
            load_time = time.time() - start
            loss, logits, acc, summaries = self.model.test(sess, batch_data)
            run_time = time.time() - start - load_time
            self.test_writer.add_summary(summaries, step + 1 + epoch * loop_num)

            avg_loss += loss
            load_times += load_time
            run_times += run_time
            avg_acc += acc
            if (step + 1) % (self.display*1000) == 0:
              logging.info('epoch {}-test step {}: loss: {:.3f}, acc: {:.3f} in {:.3f}s, load {:.3f}s'.format(
                epoch+1, step + 1, avg_loss / self.display, avg_acc / self.display, run_times, load_times))
              load_times, run_times, avg_acc, avg_loss = 0.0, 0.0, 0.0, 0.0
              #break
            for i in range(self.batch_size):
              if pred_dict.get(batch_data[0][i]) is None:
                pred_dict[batch_data[0][i]] = []
              pred_dict[batch_data[0][i]].append([logits[i], int(batch_data[-2][i]),int(batch_data[-1][i])])

          precision, recall, ndcg, auc = evaluation(pred_dict, 0)
          logging.info('test auc: {:.4f}, ndcg: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
            auc, ndcg, precision, recall))
          save_path = os.path.join(self.model_dir, 'model-{:.4f}-{:.4f}.ckpt'.format(auc, min_loss))
          self.model.save(sess, save_path, epoch + 1)
          break


  def test(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    load_times, run_times = 0.0, 0.0
    with tf.Session(config=config) as sess:
      self.create_model(sess)
      logging.info('start test phase')
      test_epoch_length = self.data.epoch_test_length
      loop_num = int(test_epoch_length // self.batch_size)
      logging.info('test iterations: {}'.format(loop_num))
      pred_dict = {}
      for step in range(loop_num):
        start = time.time()
        batch_data = self.data.get_test_batch(step)
        load_time = time.time() - start
        _, logits, _, _ = self.model.test(sess, batch_data)
        run_time = time.time() - start - load_time

        load_times += load_time
        run_times += run_time
        if (step + 1) % (self.display * 100) == 0:
          logging.info('test step {}: in {:.3f}s, load {:.3f}s'.format(
            step + 1, run_times, load_times))
          load_times, run_times, avg_acc, avg_loss = 0.0, 0.0, 0.0, 0.0
          #break
        for i in range(self.batch_size):
          if pred_dict.get(batch_data[0][i]) is None:
            pred_dict[batch_data[0][i]] = []
          pred_dict[batch_data[0][i]].append([logits[i], int(batch_data[-2][i]), int(batch_data[-1][i])])

      pres, recalls, ndcgs = [], [], []
      for k in range(1, 11):
        top_k = k * 10
        precision, recall, ndcg, auc = evaluation(pred_dict, top_k)
        logging.info('test auc: {:.4f}, ndcg: {:.4f}, precision: {:.4f}, recall: {:.4f} in top {}'.format(
          auc, ndcg, precision, recall, top_k))
        pres.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
      pres_str = ['{:.4f}'.format(v) for v in pres]
      recalls_str = ['{:.4f}'.format(v) for v in recalls]
      ndcgs_str = ['{:.4f}'.format(v) for v in ndcgs]
      logging.info('precisions:{}'.format(','.join(pres_str)))
      logging.info('recalls: {}'.format(','.join(recalls_str)))
      logging.info('ndcgs:{}'.format(','.join(ndcgs_str)))

def evaluation(pred_dict, top_k=0):
  if top_k == 0:
    flag = True
  else:
    flag = False
  precisions, recalls, ndcgs, auc_lst = [], [], [], []
  for key in pred_dict:
    preds = pred_dict[key]
    preds.sort(key=lambda x: x[0], reverse=True)
    preds = np.array(preds)
    pos_num = sum(preds[:, 1])
    neg_num = len(preds) - pos_num
    if pos_num == 0 or neg_num == 0:
      continue
    if flag:
      top_k = len(preds)
    # precision and recall
    precisions.append([sum(preds[:top_k, 1]) / top_k, len(preds)])
    recalls.append([sum(preds[:top_k, 1]) / sum(preds[:, 1]), len(preds)])

    # ndcg
    pos_idx = np.where(preds[:top_k, 1] == 1)[0]
    dcg = np.sum(np.log(2) / np.log(2 + pos_idx))
    idcg = np.sum(np.log(2) / np.log(2 + np.arange(len(pos_idx))))
    ndcg = dcg / (idcg + 1e-8)
    ndcgs.append([ndcg, len(preds[:top_k])])

    # auc
    pos_count, neg_count = 0, 0
    for i in range(len(preds)):
      if preds[i, 1] == 0:
        neg_count += (pos_num - pos_count)
      else:
        pos_count += 1
      if pos_count == pos_num:
        auc = 1 - (neg_count / (pos_num * neg_num))
        auc_lst.append([auc, len(preds), key])
        break

  precisions = np.array(precisions)
  p = sum(precisions[:, 0] * precisions[:, 1]) / sum(precisions[:, 1])
  recalls = np.array(recalls)
  r = sum(recalls[:, 0] * recalls[:, 1]) / sum(recalls[:, 1])
  ndcgs = np.array(ndcgs)
  ndcg = sum(ndcgs[:, 0] * ndcgs[:, 1]) / sum(ndcgs[:, 1])
  auc_lst = np.array(auc_lst)
  auc = sum(auc_lst[:, 0] * auc_lst[:, 1]) / sum(auc_lst[:, 1])

  return p, r, ndcg, auc