#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import matlab.engine
import segy_read_write
import tensorflow as tf
from model import *
from utils import *
import PIL
from config import configPara
os.environ["CUDA_VISIBLE_DEVICES"]="0"


print("---------------------------defining model---------------------------")
# including 2 generators and 2 discriminators
nn1 = configPara.nn1
AI_H_real = tf.placeholder('float32', [configPara.batch_size, nn1, nn1, 1], name='AI_H_real')
AI_L_real = tf.placeholder('float32', [configPara.batch_size, nn1, nn1, 1], name='AI_L_real')
S_real = tf.placeholder('float32', [configPara.batch_size, nn1, nn1, 1], name='S_real')

S_generate_mid = generator(AI_H_real+AI_L_real, reuse=False, net_name="F_net")
AI_H_generate = generator(S_generate_mid, reuse=False, net_name="I_net")

AI_H_generate_mid = generator(S_real, reuse=True, net_name="I_net")
S_generate = generator(AI_H_generate_mid+AI_L_real, reuse=True, net_name="F_net")

d1_prob_fake, d1_logit_fake = discriminator(AI_H_generate_mid, net_name="D1")
d1_prob_real, d1_logit_real = discriminator(AI_H_real, reuse=True, net_name="D1")


d2_prob_fake, d2_logit_fake = discriminator(S_generate_mid, net_name="D2")
d2_prob_real, d2_logit_real = discriminator(S_real, reuse=True,net_name="D2")

# loss 1
with tf.variable_scope('loss'):
    with tf.variable_scope('d_loss'):
        d1_loss = -tf.reduce_sum(tf.log(d1_prob_real+1e-12) + tf.log(1. - d1_prob_fake+1e-12)) / configPara.batch_size
        d1_pred_fake = tf.cast(tf.round(d1_prob_fake), tf.float32)
        d1_prob_real = tf.cast(tf.round(d1_prob_real), tf.float32)
        d1_accuracy = (tf.reduce_sum(tf.cast(tf.equal(d1_pred_fake, tf.zeros([configPara.batch_size, 1])), tf.float32))
                       + tf.reduce_sum(tf.cast(tf.equal(d1_prob_real, tf.ones([configPara.batch_size, 1])),
                                               tf.float32))) / configPara.batch_size / 2.0
        d2_loss = -tf.reduce_sum(
            tf.log(d2_prob_real + 1e-12) + tf.log(1. - d2_prob_fake + 1e-12)) / configPara.batch_size
        d2_pred_fake = tf.cast(tf.round(d2_prob_fake), tf.float32)
        d2_prob_real = tf.cast(tf.round(d2_prob_real), tf.float32)
        d2_accuracy = (tf.reduce_sum(tf.cast(tf.equal(d2_pred_fake, tf.zeros([configPara.batch_size, 1])), tf.float32))
                       + tf.reduce_sum(tf.cast(tf.equal(d2_prob_real, tf.ones([configPara.batch_size, 1])),
                                               tf.float32))) / configPara.batch_size / 2.0
        d_loss = d1_loss + d2_loss
    with tf.variable_scope('g_loss'):
        AI_generate_loss = tf.reduce_mean(tf.abs(AI_H_generate - AI_H_real))               # L1 loss
        AI_discriminate_loss = -tf.reduce_sum(tf.log(d1_prob_fake+1e-12)) / configPara.batch_size

        S_generate_loss = tf.reduce_mean(tf.abs(S_generate - S_real))                      # L1 loss
        S_discriminate_loss = -tf.reduce_sum(tf.log(d2_prob_fake+1e-12)) / configPara.batch_size
        # loss all
        AI_loss = configPara.lambda_AI * AI_generate_loss + AI_discriminate_loss
        S_loss = configPara.lambda_S * S_generate_loss + S_discriminate_loss

        g_loss = AI_loss + S_loss

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(configPara.lr_init, trainable=False)

train_vars = tf.trainable_variables()
theta_d1 = [var for var in train_vars if 'D1' in var.name]
theta_d2 = [var for var in train_vars if 'D2' in var.name]
theta_d = theta_d1 + theta_d2
d_solver = tf.train.AdamOptimizer(lr_v, beta1=configPara.beta1).minimize(d_loss, var_list=theta_d)

theta_F = [var for var in train_vars if 'F_net' in var.name]
theta_I = [var for var in train_vars if 'I_net' in var.name]
theta_g = theta_F + theta_I
with tf.variable_scope('adam'):
    g_solver = tf.train.AdamOptimizer(lr_v, beta1=configPara.beta1).minimize(g_loss, var_list=theta_g)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())
start_epoch = 0
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_summary/',
                                         sess.graph)
