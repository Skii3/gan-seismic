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
S_generate_mid_target = tf.placeholder('float32', [configPara.batch_size, nn1, nn1, 1], name='S_generate_mid_target')
AI_H_generate = generator(S_generate_mid, reuse=False, net_name="I_net")

AI_H_generate_mid = generator(S_real, reuse=True, net_name="I_net")
AI_H_generate_mid_target = tf.placeholder('float32', [configPara.batch_size, nn1, nn1, 1],
                                          name='AI_H_generate_mid_target')
S_generate = generator(AI_H_generate_mid+AI_L_real, reuse=True, net_name="F_net")
d1_prob_fake, d1_logit_fake = discriminator(AI_H_generate_mid, net_name="D1")
d1_prob_real, d1_logit_real = discriminator(AI_H_real, reuse=True, net_name="D1")


d2_prob_fake, d2_logit_fake = discriminator(S_generate_mid, net_name="D2")
d2_prob_real, d2_logit_real = discriminator(S_real, reuse=True,net_name="D2")

# loss 1
d1_loss = -tf.reduce_sum(tf.log(d1_prob_real+1e-12) + tf.log(1. - d1_prob_fake+1e-12)) / configPara.batch_size
AI_generate_loss = tf.reduce_mean(tf.abs(AI_H_generate - AI_H_real))               # L1 loss
S_generate_mid_loss = tf.reduce_mean(tf.abs(S_generate_mid - S_generate_mid_target))
AI_discriminate_loss = -tf.reduce_sum(tf.log(d1_prob_fake+1e-12)) / configPara.batch_size
d1_pred_fake = tf.cast(tf.round(d1_prob_fake), tf.float32)
d1_prob_real = tf.cast(tf.round(d1_prob_real), tf.float32)
d1_accuracy = (tf.reduce_sum(tf.cast(tf.equal(d1_pred_fake, tf.zeros([configPara.batch_size, 1])), tf.float32))
               + tf.reduce_sum(tf.cast(tf.equal(d1_prob_real, tf.ones([configPara.batch_size, 1])), tf.float32))) / configPara.batch_size / 2.0

d2_loss = -tf.reduce_sum(tf.log(d2_prob_real+1e-12) + tf.log(1. - d2_prob_fake+1e-12)) / configPara.batch_size
S_generate_loss = tf.reduce_mean(tf.abs(S_generate - S_real))                      # L1 loss
AI_H_generate_mid_loss = tf.reduce_mean(tf.abs(AI_H_generate_mid-AI_H_generate_mid_target))
S_discriminate_loss = -tf.reduce_sum(tf.log(d2_prob_fake+1e-12)) / configPara.batch_size
d2_pred_fake = tf.cast(tf.round(d2_prob_fake), tf.float32)
d2_prob_real = tf.cast(tf.round(d2_prob_real), tf.float32)
d2_accuracy = (tf.reduce_sum(tf.cast(tf.equal(d2_pred_fake, tf.zeros([configPara.batch_size, 1])), tf.float32))
               + tf.reduce_sum(tf.cast(tf.equal(d2_prob_real, tf.ones([configPara.batch_size, 1])), tf.float32))) / configPara.batch_size / 2.0

# loss all
AI_loss = configPara.lambda_AI * AI_generate_loss + AI_discriminate_loss
S_loss = configPara.lambda_S * S_generate_loss + S_discriminate_loss
d_loss = d1_loss + d2_loss
g_loss = AI_loss + S_loss + configPara.lambda_AI_H_generate_mid_loss * AI_H_generate_mid_loss + \
            configPara.lambda_S_generate_mid_loss * S_generate_mid_loss

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
g_solver = tf.train.AdamOptimizer(lr_v, beta1=configPara.beta1).minimize(g_loss, var_list=theta_g)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())
start_epoch = 0
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_summary/',
                                         sess.graph)
print("---------------------------restoring model---------------------------")
if configPara.if_continue_train:
    checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
    print checkpoint
    tf.train.Saver().restore(sess, checkpoint)
    start_epoch = int(checkpoint[checkpoint.find('model')+5:])

def train():
    print(configPara)
    print("---------------------------reading image---------------------------")

    train_img_S, index_min, index_max, max_save_S = read_all_imgs(configPara.train_image_S,1)   # 600*735
    temp = len(str(int(round(max_save_S))))
    max_save_S = np.ceil(max_save_S / pow(10, temp-1)) * pow(10, temp-1)
    train_img_S /= max_save_S
    train_img_S = expand_imgs(train_img_S)

    train_img_AI_H, max_save_AI_H = read_all_imgs(configPara.train_image_AI_H, 0)
    train_img_AI_L, max_save_AI_L = read_all_imgs(configPara.train_image_AI_L, 0)
    train_img_AI_H = train_img_AI_H[index_min:index_max, :]
    train_img_AI_L = train_img_AI_L[index_min:index_max, :]
    train_img_AI = train_img_AI_L + train_img_AI_H
    max_save_AI = np.max(np.abs(train_img_AI))
    temp = len(str(int(round(max_save_AI))))
    max_save_AI = np.ceil(max_save_AI / pow(10, temp - 1)) * pow(10, temp - 1)
    train_img_AI_H /= max_save_AI
    train_img_AI_L /= max_save_AI
    #train_img_AI /= max_save_AI
    train_img_AI_H = expand_imgs(train_img_AI_H)
    train_img_AI_L = expand_imgs(train_img_AI_L)
    #train_img_AI = expand_imgs(train_img_AI)


    test_imgs_S,_,_,_ = read_all_imgs(configPara.test_image_path, 1)
    test_imgs_S = test_imgs_S / max_save_S
    print("---------------------------training model---------------------------")
    dataNum_S = len(train_img_S)
    dataNum_AI_H = len(train_img_AI_H)
    n_epoch = configPara.n_epoch
    seq_AI_H = np.arange(0, dataNum_AI_H)
    seq_S = np.arange(0, dataNum_S)
    iter_all = 50

    loss_list_name = ['d1_loss', 'd2_loss', 'd1_accuracy', 'd2_accuracy',
                      'AI_discriminate_loss', 'S_discriminate_loss',
                      'AI_generate_loss', 'S_generate_loss',
                      'AI_loss', 'S_loss',
                      'AI_generate_mid_loss', 'S_generate_mid_loss']
    #loss_list_tensor = [d1_loss, d2_loss, AI_discriminate_loss, AI_loss,
    #                    AI_generate_loss, S_discriminate_loss, S_loss,
    #                    S_generate_loss, S_generate_mid_loss, AI_H_generate_mid_loss,
    #                    d1_accuracy, d2_accuracy]
    loss_list_epoch_all = np.zeros([len(loss_list_name)])
    for epoch in range(start_epoch, n_epoch + 1):
        d_accuracy_iter_all_last = (loss_list_epoch_all[loss_list_name.index('d1_accuracy')]
                                    + loss_list_epoch_all[loss_list_name.index('d2_accuracy')])/2
        loss_list_epoch_all = np.zeros([len(loss_list_name)])
        epoch_time = time.time()
        if (epoch % configPara.save_model_freq == 0):
            print('[*]saving model')
            tf.train.Saver().save(sess, configPara.checkpoint_dir + '/model%d' % epoch)
        if (epoch % configPara.show_freq == 0):
            imgs_input_AI_H_feed, imgs_input_AI_L_feed, imgs_target_S_feed, \
            imgs_input_S_feed, imgs_target_AI_H_feed, imgs_target_AI_L_feed\
                =extract_feed_data(seq_AI_H, train_img_AI_H, train_img_AI_L, train_img_S, seq_S)
            imgs_input_AI_feed = imgs_input_AI_L_feed + imgs_input_AI_H_feed
            S_fake, AI_H_fake_mid = sess.run([S_generate, AI_H_generate_mid],
                                             feed_dict={S_real: imgs_input_S_feed, AI_L_real: imgs_input_AI_L_feed})
            AI_H_fake, S_fake_mid = sess.run([AI_H_generate, S_generate_mid],
                                 feed_dict={AI_H_real: imgs_input_AI_H_feed, AI_L_real: imgs_input_AI_L_feed})

            residual1 = S_fake[0].squeeze() - imgs_input_S_feed[0].squeeze()
            residual2 = AI_H_fake_mid[0].squeeze() - imgs_target_AI_H_feed[0].squeeze()
            result_S = np.concatenate((S_fake[0].squeeze(), imgs_input_S_feed[0].squeeze(),residual1,
                                     AI_H_fake_mid[0].squeeze(), imgs_target_AI_H_feed[0].squeeze(),residual2),axis=1)
            residual1 = AI_H_fake[0].squeeze() - imgs_input_AI_H_feed[0].squeeze()
            residual2 = S_fake_mid[0].squeeze() - imgs_target_S_feed[0].squeeze()
            result_AI = np.concatenate((AI_H_fake[0].squeeze(), imgs_input_AI_H_feed[0].squeeze(),
                                        imgs_input_AI_feed[0].squeeze(),residual1,
                                        S_fake_mid[0].squeeze(), imgs_target_S_feed[0].squeeze(),residual2),axis=1)

            for i in range(9):
                imgs_input_AI_H_feed, imgs_input_AI_L_feed, imgs_target_S_feed, \
                imgs_input_S_feed, imgs_target_AI_H_feed, imgs_target_AI_L_feed \
                    = extract_feed_data(seq_AI_H, train_img_AI_H, train_img_AI_L, train_img_S, seq_S)
                imgs_input_AI_feed = imgs_input_AI_L_feed + imgs_input_AI_H_feed
                S_fake, AI_H_fake_mid = sess.run([S_generate, AI_H_generate_mid],
                                                 feed_dict={S_real: imgs_input_S_feed, AI_L_real: imgs_input_AI_L_feed})
                AI_H_fake, S_fake_mid = sess.run([AI_H_generate, S_generate_mid],
                                                 feed_dict={AI_H_real: imgs_input_AI_H_feed,
                                                            AI_L_real: imgs_input_AI_L_feed})

                residual1 = S_fake[0].squeeze() - imgs_input_S_feed[0].squeeze()
                residual2 = AI_H_fake_mid[0].squeeze() - imgs_target_AI_H_feed[0].squeeze()
                out_S = np.concatenate((S_fake[0].squeeze(), imgs_input_S_feed[0].squeeze(), residual1,
                                           AI_H_fake_mid[0].squeeze(), imgs_target_AI_H_feed[0].squeeze(), residual2),
                                          axis=1)
                residual1 = AI_H_fake[0].squeeze() - imgs_input_AI_H_feed[0].squeeze()
                residual2 = S_fake_mid[0].squeeze() - imgs_target_S_feed[0].squeeze()
                out_AI = np.concatenate((AI_H_fake[0].squeeze(), imgs_input_AI_H_feed[0].squeeze(),
                                            imgs_input_AI_feed[0].squeeze(), residual1,
                                            S_fake_mid[0].squeeze(), imgs_target_S_feed[0].squeeze(), residual2),
                                           axis=1)

                result_S = np.concatenate((result_S, out_S), axis=0)
                result_AI = np.concatenate((result_AI, out_AI), axis=0)

            scipy.misc.imsave(configPara.samples_save_dir + '/train_forward_%d_S.png' % epoch, result_S)
            scipy.misc.imsave(configPara.samples_save_dir + '/train_forward_%d_AI.png' % epoch, result_AI)
        loss_list_iter_all = np.zeros([len(loss_list_name)])
        for idx in range(0, iter_all, 1):
            ############################################# train d and generator #############################################
            imgs_real_AI_H_feed, imgs_input_AI_L_feed, imgs_target_S_feed, imgs_real_S_feed, imgs_target_AI_H_feed, _ \
                = extract_feed_data(seq_AI_H, train_img_AI_H, train_img_AI_L, train_img_S, seq_S)
            n_repeat = configPara.repeat_d
            if d_accuracy_iter_all_last < 0.2 and epoch>0:
                n_repeat = n_repeat * 2
            temp_d = np.zeros(4)
            for repeat in range(n_repeat):
                d_loss_eval \
                    = sess.run([d_solver, d1_loss, d1_accuracy, d2_loss, d2_accuracy],
                               feed_dict={AI_H_real: imgs_real_AI_H_feed, S_real: imgs_real_S_feed,
                                          AI_L_real: imgs_input_AI_L_feed})
                temp_d = temp_d + d_loss_eval[1:]
            temp_d /= n_repeat
            ind_d = [loss_list_name.index('d1_loss'), loss_list_name.index('d1_accuracy'),
                     loss_list_name.index('d2_loss'), loss_list_name.index('d2_accuracy')]
            n_repeat = configPara.repeat_g
            if d_accuracy_iter_all_last >= 0.9 and epoch > 0:
                n_repeat = n_repeat * 2
            temp_g = np.zeros(8)
            for repeat in range(n_repeat):
                g_loss_eval \
                    = sess.run(
                    [g_solver, AI_discriminate_loss, AI_generate_loss, AI_loss, S_discriminate_loss, S_generate_loss,
                     S_loss, AI_H_generate_mid_loss, S_generate_mid_loss],
                    feed_dict={S_real: imgs_real_S_feed, AI_H_real: imgs_real_AI_H_feed,
                               AI_L_real: imgs_input_AI_L_feed, S_generate_mid_target: imgs_target_S_feed,
                               AI_H_generate_mid_target: imgs_target_AI_H_feed})
                temp_g = temp_g + g_loss_eval[1:]
            temp_g /= n_repeat
            ind_g = [loss_list_name.index('AI_discriminate_loss'), loss_list_name.index('AI_generate_loss'),
                     loss_list_name.index('AI_loss'), loss_list_name.index('S_discriminate_loss'),
                     loss_list_name.index('S_generate_loss'), loss_list_name.index('S_loss'),
                     loss_list_name.index('AI_generate_mid_loss'), loss_list_name.index('S_generate_mid_loss')]

            loss_list_iter_all[ind_d] = temp_d
            loss_list_iter_all[ind_g] = temp_g
            loss_list_epoch_all += loss_list_iter_all
            if idx % configPara.show_iter == 0:
                show_str = "[*] iter [%2d/%2d]\n" % (idx, iter_all)
                for i_name in range(len(loss_list_name)/2):
                    ind_name = i_name * 2
                    show_str += "  %20s: %.4f\t\t%20s: %.4f\n" % (loss_list_name[ind_name],loss_list_iter_all[ind_name],
                                                            loss_list_name[ind_name+1], loss_list_iter_all[ind_name+1])
                print(show_str)
        print('*********************************************************************************************')
        loss_list_epoch_all /= iter_all
        show_str = "[*] Epoch [%2d/%2d] time: %4.4fs\n" % (epoch, n_epoch, time.time()-epoch_time)
        for i_name in range(len(loss_list_name) / 2):
            ind_name = i_name * 2
            show_str += "  %20s: %.4f\t\t%20s: %.4f\n" % (loss_list_name[ind_name], loss_list_epoch_all[ind_name],
                                                          loss_list_name[ind_name + 1], loss_list_epoch_all[ind_name + 1])
        print(show_str)
        if epoch % configPara.test_freq == 0:
            if configPara.type != 0:
                print 'error test file path!!!'
            print("---------------------------test--------------------------------")
            runTest(test_imgs_S, sess, epoch)
            print('Done!')

def runTest(test_imgs, sess, epoch):
    test_imgs_all = []
    test_imgs_all.append(test_imgs)
    inputImage = test_imgs_all[0]
    w, h = inputImage.shape
    n = 8
    h2 = h - h % n
    w2 = w - w % n
    S_input_test = tf.placeholder('float32', [1, w2, h2, 1], name='S_input_test')
    AI_H_generate_mid_test = generator(S_input_test, reuse=True, net_name="I_net")
    #S_generate_test = generator(AI_H_generate_mid_test + AI_L_real, reuse=True, net_name="F_net")

    for i in range(0, len(test_imgs_all), 1):
        inputImage = test_imgs_all[i]
        w, h = inputImage.shape

        n = 8
        h2 = h - h % n
        w2 = w - w % n
        imgs_input_S_feed = np.expand_dims(np.expand_dims(inputImage[:w2,:h2],axis=0),axis=3)
        AI_H_fake = sess.run(AI_H_generate_mid_test, feed_dict={S_input_test: imgs_input_S_feed})

        scipy.misc.imsave(configPara.test_save_dir + '/%d_%d_test_S.png' % (epoch, i), imgs_input_S_feed.squeeze())
        #scipy.misc.imsave(configPara.test_save_dir + '/%d_%d_test_S_fake.png' % (epoch, i), reScale(S_fake.squeeze()))
        scipy.misc.imsave(configPara.test_save_dir + '/%d_%d_test_AI_H_fake.png' % (epoch, i), AI_H_fake.squeeze())

def test():
    print(configPara)
    print("---------------------------reading image---------------------------")

    train_img_S, index_min, index_max, max_save_S = read_all_imgs(configPara.train_image_S, 1)  # 600*735
    temp = len(str(int(round(max_save_S))))
    max_save_S = np.ceil(max_save_S / pow(10, temp - 1)) * pow(10, temp - 1)
    train_img_S /= max_save_S
    train_img_S = expand_imgs(train_img_S)

    train_img_AI_H, max_save_AI_H = read_all_imgs(configPara.train_image_AI_H, 0)
    train_img_AI_L, max_save_AI_L = read_all_imgs(configPara.train_image_AI_L, 0)
    train_img_AI_H = train_img_AI_H[index_min:index_max, :]
    train_img_AI_L = train_img_AI_L[index_min:index_max, :]
    train_img_AI = train_img_AI_L + train_img_AI_H
    max_save_AI = np.max(np.abs(train_img_AI))
    temp = len(str(int(round(max_save_AI))))
    max_save_AI = np.ceil(max_save_AI / pow(10, temp - 1)) * pow(10, temp - 1)
    train_img_AI_H /= max_save_AI
    train_img_AI_L /= max_save_AI
    # train_img_AI /= max_save_AI
    train_img_AI_H = expand_imgs(train_img_AI_H)
    train_img_AI_L = expand_imgs(train_img_AI_L)
    # train_img_AI = expand_imgs(train_img_AI)

    test_imgs_S, _, _, _ = read_all_imgs(configPara.test_image_path, 1)
    test_imgs_S = test_imgs_S / max_save_S
    runTest(test_imgs_S, sess, start_epoch)

if configPara.type == 0:
    train()
    os.system('cp ./console_log.txt ./train_result/train_log'+'_'+str(start_epoch)+'.txt')
else:
    test()