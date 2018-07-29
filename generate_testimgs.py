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

def main():
    test_imgs = read_all_imgs("test_data/plut_target/", regx='*.txt')
    #print(len(test_imgs))
    test_imgs = generate_noiseimgs(test_imgs,20)

if __name__ == '__main__':
    main()
