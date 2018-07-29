# -*- coding: utf8 -*-
from easydict import EasyDict as edict
import json
import numpy as np
import os

configPara = edict()

configPara.if_continue_train=False
configPara.test=True

configPara.nn1=128
configPara.nn2=128
configPara.rate=4

configPara.lr_init = 2e-4
configPara.beta1 = 0.5
configPara.n_epoch =500

configPara.L1_lambda=1
configPara.tvDiff_lambda=200
configPara.loop_lambda=1
configPara.gan_lambda=0.01

configPara.test_freq=20
configPara.save_model_freq=100
configPara.show_freq=10
configPara.show_iter = 10

configPara.if_aug = True

if configPara.if_aug:
    configPara.scale = 5
else:
    configPara.scale = 0
    
configPara.if_scale = True

configPara.batch_size = 10
nn1=configPara.nn1
nn2=configPara.nn2
scale=configPara.scale
tvDiff_lambda=configPara.tvDiff_lambda
configPara.type = 0
TYPE = configPara.type          # 0 : 训练 ，  1: 测试plut   ， 2：测试sigsbee  3: 测试plut 15  ， 4：测试sigsbee20
pre = "train_result/"
configPara.samples_save_dir = pre+"samples/"
configPara.checkpoint_dir = pre+"checkpoint/"
configPara.buffer_dir = pre+"buffer/"

configPara.test_save_dir = pre+"test_result"           # 训练用
configPara.train_image_S = "data/synthetic/seismic_syn_wavelet.sgy"
configPara.train_image_AI_H = "data/synthetic/imp_high.sgy"
configPara.train_image_AI_L = "data/synthetic/imp_low.sgy"
configPara.test_image_path="data/real/well_seis_to_lu.sgy"
configPara.repeat_d = 1
configPara.repeat_g = 2
configPara.lambda_AI = 100
configPara.lambda_S = 100
configPara.lambda_AI_H_generate_mid_loss = 200
configPara.lambda_S_generate_mid_loss = 200


if not os.path.exists(configPara.checkpoint_dir):
    os.makedirs(configPara.checkpoint_dir)
if not os.path.exists(configPara.test_save_dir):
    os.makedirs(configPara.test_save_dir)
if not os.path.exists(configPara.samples_save_dir):
    os.makedirs(configPara.samples_save_dir)
if not os.path.exists(configPara.buffer_dir):
    os.makedirs(configPara.buffer_dir)
