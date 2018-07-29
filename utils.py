import tensorflow as tf
import PIL
import scipy.misc
import numpy as np
import re
import os
import random
from config import configPara
from skimage.util import random_noise
from scipy import ndimage
import segy_read_write

def read_all_imgs(path,type=0):
    f = segy_read_write.open_segy_file(path)
    data = segy_read_write.read_partially(f, 0, f.tracecount+1)
    max_save = np.max(np.abs(data))
    if type == 1:
        index = np.sum(data, axis=1)
        index[index != 0] = 1
        index = np.cumsum(index)
        temp = index[-1]
        index = np.where(index != 0)
        index_min = index[0][0]
        index_max = index_min + temp
        data = data[int(index_min):int(index_max),:]
        return data, int(index_min), int(index_max), max_save
    elif type == 0:
        return data, max_save

def expand_imgs(imgs):
    imgs_temp = imgs
    imgs = []
    imgs.append(imgs_temp)
    for k in range(0,len(imgs)):
        img=imgs[k]
        h,w = img.shape
        for step in range(2,configPara.scale+1):

            temp=img[0:h,0:w:step]
            imgs.append(temp)
    for k in range(0,len(imgs)):
        scipy.misc.imsave(configPara.buffer_dir+'/train_%d.png' % k,imgs[k])
    return imgs

def sampleImg(inputImg,nn):
    w,h=inputImg.shape

    h_size=nn
    w_size=nn
    if h-h_size <= 0 or w-w_size <= 0:
        return None,None
    h_loc=random.randint(0, h-h_size)
    w_loc=random.randint(0, w-w_size)
    inputImage=inputImg[w_loc:w_loc+w_size,h_loc:h_loc+h_size]

    aug_type=random.randint(0, 12)
    if aug_type==1:
        inputImage=np.fliplr(inputImage);
    if aug_type==2:
        inputImage=np.flipud(inputImage);
    if aug_type==3:
        inputImage=np.rot90(inputImage,1);
    if aug_type==4:
        inputImage=np.rot90(inputImage,2);
    if aug_type==5:
        inputImage=np.rot90(inputImage,3);
    if aug_type==6:
        inputImage=np.flipud(np.rot90(inputImage,1));
    if aug_type==7:
        inputImage=np.flipud(np.rot90(inputImage,2));
    if aug_type==8:
        inputImage=np.flipud(np.rot90(inputImage,3));
    if aug_type==9:
        inputImage=np.fliplr(np.rot90(inputImage,1));
    if aug_type==10:
        inputImage=np.fliplr(np.rot90(inputImage,2));
    if aug_type==11:
        inputImage=np.fliplr(np.rot90(inputImage,3));
    '''
    intensity_aug=random.randint(1, 5)
    if intensity_aug==2:
        inputImage = inputImage * np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)
    if intensity_aug==3:
        inputImage = inputImage / np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)
    '''

    return inputImage

def extract_feed_data(seq_AI_H,train_img_AI_H,train_img_AI_L,train_img_S,seq_S):
    nn1 = configPara.nn1
    seq_AI_H = np.random.permutation(seq_AI_H)
    imgs_input_AI_H = train_img_AI_H[seq_AI_H[0]]
    imgs_input_AI_H_feed = []
    imgs_input_AI_L = train_img_AI_L[seq_AI_H[0]]
    imgs_input_AI_L_feed = []
    imgs_input_S = train_img_S[seq_AI_H[0]]
    imgs_target_S_feed = []
    for ii in range(configPara.batch_size):
        imgs_input_AI_H_temp, imgs_input_AI_L_temp, imgs_target_S \
            = sampleImg_multi([imgs_input_AI_H, imgs_input_AI_L, imgs_input_S], configPara.nn1)
        imgs_input_AI_H_feed.append(imgs_input_AI_H_temp)
        imgs_input_AI_L_feed.append(imgs_input_AI_L_temp)
        imgs_target_S_feed.append(imgs_target_S)
    imgs_input_AI_H_feed = np.reshape(imgs_input_AI_H_feed, [configPara.batch_size, nn1, nn1, 1])
    imgs_input_AI_L_feed = np.reshape(imgs_input_AI_L_feed, [configPara.batch_size, nn1, nn1, 1])
    imgs_target_S_feed = np.reshape(imgs_target_S_feed, [configPara.batch_size, nn1, nn1, 1])

    seq_S = np.random.permutation(seq_S)
    imgs_input_S = train_img_S[seq_S[0]]
    imgs_input_S_feed = []
    imgs_input_AI_H = train_img_AI_H[seq_S[0]]
    imgs_input_AI_L = train_img_AI_L[seq_S[0]]
    imgs_target_AI_H_feed = []
    imgs_target_AI_L_feed = []
    for ii in range(configPara.batch_size):
        imgs_input_S_temp, imgs_target_AI_H_temp, imgs_target_AI_L_temp \
            = sampleImg_multi([imgs_input_S, imgs_input_AI_H, imgs_input_AI_L], configPara.nn1)
        imgs_input_S_feed.append(imgs_input_S_temp)
        imgs_target_AI_H_feed.append(imgs_target_AI_H_temp)
        imgs_target_AI_L_feed.append(imgs_target_AI_L_temp)
    imgs_input_S_feed = np.reshape(imgs_input_S_feed, [configPara.batch_size, nn1, nn1, 1])
    imgs_target_AI_H_feed = np.reshape(imgs_target_AI_H_feed, [configPara.batch_size, nn1, nn1, 1])
    imgs_target_AI_L_feed = np.reshape(imgs_target_AI_L_feed, [configPara.batch_size, nn1, nn1, 1])
    return imgs_input_AI_H_feed, imgs_input_AI_L_feed, imgs_target_S_feed, \
           imgs_input_S_feed, imgs_target_AI_H_feed, imgs_target_AI_L_feed

def sampleImg_multi(inputImg,nn):
    n_input = len(inputImg)

    w,h=inputImg[0].shape

    h_size=nn
    w_size=nn
    if h-h_size <= 0 or w-w_size <= 0:
        return None,None
    h_loc=random.randint(0, h-h_size)
    w_loc=random.randint(0, w-w_size)
    inputImage = []
    for i in range(n_input):
        inputImage.append(inputImg[i][w_loc:w_loc+w_size,h_loc:h_loc+h_size])
    aug_type=random.randint(0, 12)
    if aug_type==1:
        for i in range(n_input):
            inputImage[i] = np.fliplr(inputImage[i])
    if aug_type==2:
        for i in range(n_input):
            inputImage[i] = np.flipud(inputImage[i])
    if aug_type==3:
        for i in range(n_input):
            inputImage[i] = np.rot90(inputImage[i], 1)
    if aug_type==4:
        for i in range(n_input):
            inputImage[i] = np.rot90(inputImage[i], 2)
    if aug_type==5:
        for i in range(n_input):
            inputImage[i] = np.rot90(inputImage[i], 3)
    if aug_type==6:
        for i in range(n_input):
            inputImage[i] = np.flipud(np.rot90(inputImage[i], 1))
    if aug_type==7:
        for i in range(n_input):
            inputImage[i] = np.flipud(np.rot90(inputImage[i], 2))
    if aug_type==8:
        for i in range(n_input):
            inputImage[i] = np.flipud(np.rot90(inputImage[i], 3))
    if aug_type==9:
        for i in range(n_input):
            inputImage[i] = np.fliplr(np.rot90(inputImage[i], 1))
    if aug_type==10:
        for i in range(n_input):
            inputImage[i] = np.fliplr(np.rot90(inputImage[i], 2))
    if aug_type==11:
        for i in range(n_input):
            inputImage[i] = np.fliplr(np.rot90(inputImage[i], 3))
    '''
    intensity_aug=random.randint(1, 5)
    if intensity_aug==2:
        inputImage = inputImage * np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)
    if intensity_aug==3:
        inputImage = inputImage / np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)
    '''

    return inputImage

def downsample(x):
    [w,h,c]=x.shape
    rate=configPara.rate
    x =  scipy.misc.imresize(x, size=[w/rate/2, h/rate/2], interp='bicubic', mode=None)
    x =  scipy.misc.imresize(x, size=[w, h], interp='bicubic', mode=None)
    return x

def myScale(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def reScale(x):
    x = x+1
    x = x*(255. / 2.)
    return x

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imageDiff(img1,img2):
    pos = tf.constant(np.identity(3), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(img1, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(img1, filter_y, strides, padding=padding))

    gt_dx = tf.abs(tf.nn.conv2d(img2, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(img2, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    return grad_diff_x,grad_diff_y
