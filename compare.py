#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
from model import *
from utils import *
import PIL
from config import configPara

os.environ["CUDA_VISIBLE_DEVICES"]=""

def runTest(test_imgs,test_img,sess,path,model_type=0):
    for i in range(0, len(test_imgs), 1):
        inputImage=test_imgs[i]
        w,h=inputImage.shape
        n=8
        h2=h-h%n
        w2=w-w%n

        input_image_forward_large = tf.placeholder('float32', [1, w2, h2, 1], name='input_test')
        image_forward_large = generator(input_image_forward_large, reuse=True,net_name="net_forward")

        inputImage2=inputImage[0:w2,0:h2]
        test_img = test_img[0:w2,0:h2]
        inputImage2=np.expand_dims(np.expand_dims(inputImage2,axis=0),axis=3)
        out = sess.run(image_forward_large, {input_image_forward_large:inputImage2})
        if model_type==1:
            out=inputImage2-out
        #np.save(configPara.test_save_dir+'/my_%d.txt' % i,out.squeeze())
        np.savetxt(path+'/my_%d.txt' % i, out.squeeze(), delimiter=' ')
        np.savetxt(path+'/input_%d.txt' % i, inputImage2.squeeze(), delimiter=' ')
        np.savetxt(path+'/target_%d.txt' % i, test_img, delimiter=' ')
        #out = scipy.misc.imresize(reScale(out.squeeze()), size=[w, h], interp='bicubic', mode=None)
        #out=np.concatenate((inputImage, out),axis=1)
        scipy.misc.imsave(path+'/my_%d.png' % i,reScale(out.squeeze()))

def test():
    test_img = read_all_imgs(configPara.test_image_path, regx='.*.txt')
    test_imgs = generate_noiseimgs(test_img,20)

    input_image_forward_large = tf.placeholder('float32', [1, 512, 512, 1], name='input_test')
    image_forward_large = generator(input_image_forward_large, reuse=False,net_name="net_forward")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    print("---------------------------test1--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/1/",1)
    print("---------------------------test2--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint2/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/2/")
    print("---------------------------test3--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint3/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/3/")
    print("---------------------------test4--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint4/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/4/",1)
    print("---------------------------test5--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint5/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/5/",1)
    print("---------------------------test6--------------------------------")
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint6/")
    tf.train.Saver().restore(sess,checkpoint)
    runTest(test_imgs,test_img[0],sess,"train_result/compare/6/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    args = parser.parse_args()
    test()
