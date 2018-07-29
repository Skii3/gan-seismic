#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
from config import configPara
import tensorflow.contrib.slim as slim

def discriminator(discrim_inputs, reuse=False, net_name="discriminator"):
    ndf=64
    with tf.variable_scope(net_name, reuse=reuse) as vs:
        c1 = lrelu(conv(discrim_inputs,ndf,"c1"),0.2)
        c2 = lrelu(batchnorm(conv(c1,ndf*2,"c2"),"b2"),0.2)
        c3 = lrelu(batchnorm(conv(c2,ndf*4,"c3"),"b3"),0.2)
        c4 = lrelu(batchnorm(conv(c3,ndf*8,"c4"),"b4"),0.2)
        output_logit = denselayer(denselayer(c4,1024,"c5"),1,"c6")
        output_prob = tf.nn.sigmoid(output_logit,'prob')
        return output_prob,output_logit

def generator(generator_inputs,reuse=False,net_name="net_forward"):
    ngf=64
    with tf.variable_scope(net_name, reuse=reuse) as vs:

        c1 = conv(generator_inputs, ngf*2, "c1")
        c2 = batchnorm(conv(lrelu(c1,0.2),ngf*4,"c2"),"b2")
        c3 = batchnorm(conv(lrelu(c2,0.2),ngf*8,"c3"),"b3")

        d1 = batchnorm(deconv(tf.nn.relu(c3),ngf*4,"d1"),"b1_2")
        d2 = batchnorm(deconv(tf.nn.relu(tf.concat([d1,c2],axis=3)),ngf*2,"d2"),"b2_2")
        d3 = deconv(tf.nn.relu(tf.concat([d2,c1],axis=3)),ngf,"d3")
        output= tf.tanh(conv(tf.nn.relu(d3),1,"output",1))
        #output=tf.tanh(deconv(tf.nn.relu(tf.concat([d2,c1],axis=3)),1,"d3"))
        return output

def vector2conv(batch_input,out_channels,w_size,name):
    with tf.variable_scope(name):
        batch_size,n_in=[int(d) for d in batch_input.get_shape()]
        n_units=out_channels*w_size*w_size
        W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=tf.random_normal_initializer(0,0.02))
        matmuled=tf.matmul(batch_input, W)
        shape=[-1, w_size, w_size, out_channels]
        output=tf.reshape(matmuled, shape=shape)
        return output

def denselayer(batch_input,n_units,name):
    with tf.variable_scope(name):
        n_in = 1
        for d in batch_input.get_shape()[1:].as_list():
            n_in *= d
        reshaped=tf.reshape(batch_input, shape=[-1, n_in])
        W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=tf.random_normal_initializer(0,0.02))
        b = tf.get_variable(name='b',  initializer=tf.ones(shape=(n_units)))
        output=tf.matmul(reshaped, W) + b
        return output


def conv(batch_input, out_channels, name,stride=2):
    with tf.variable_scope(name):
        '''
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, 2, 2, 1], padding="VALID")
        return conv
        '''
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input, name):
    with tf.variable_scope(name):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def deconv(batch_input, out_channels, name):
    with tf.variable_scope(name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input,
                                      filter,
                                      [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv

def pixelShuffler(inputs, scale=2):
    with tf.variable_scope("pixelShuffler"):
        size = tf.shape(inputs)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = inputs.get_shape().as_list()[-1]

        # Get the target channel size
        channel_target = c // (scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        # Reshape and transpose for periodic shuffling for each channel
        input_split = tf.split(inputs, channel_target, axis=3)
        output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

        return output

def phaseShift(inputs, scale, shape_1, shape_2):
    with tf.variable_scope("phaseShift"):
    # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)
