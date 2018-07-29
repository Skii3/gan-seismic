#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:39:41 2018
read and write .sgy (segy) file using segyio
@author: chia
"""

import segyio
import numpy as np
#from shutil import copyfile

def open_segy_file(path='../thb_105926_label_336_337.sgy', mode='r'):
    print '[*] opening file named ' + path
    return segyio.open(path,mode)

def get_inline(f,indicator='FieldRecord'):
    print '[*] get the inline of .sgy'
    attr = {}
    inline = []
    for i, k in enumerate(f.header[0]):
        attr[str(k)] = i
    ind = attr[indicator]
    for i in range(len(f.trace)):
        inline += [int(f.header[i].items()[ind][1])]
        if i % 300000 == 0 : 
            print '[*] inline ' + str(i) + ' finished'
    return np.array(inline)

def cal_ntst_ntend(inline):
    nt = len(inline)
    dline = np.where((inline[1:] - inline[:-1]) > 0)[0]
    ntend = np.concatenate((dline, np.array([nt-1])))
    ntst = np.concatenate((np.array([0]),dline + 1))
    return ntst, ntend

def read_partially(f,st,end):
    return np.transpose(f.trace.raw[st : end + 1])

def write_partially(f,st,end,matrix):
    matrix = np.single(matrix)
    for i in range(end - st + 1):
        f.trace[st + i] = matrix[:,i]
    
def trace_norm(gather):
    for i in range(gather.shape[1]):
        trace = gather[:,i]
        if not len(trace[trace == 0]) == len(trace):
            gather[:,i] = trace / max(abs(trace))
    for j in range(gather.shape[1]):
        trace = gather[:,j]
        if len(trace[trace == 0]) == len(trace) and j > 0 and j < gather.shape[1] - 1:
            gather[:,j] = 0.5 * (gather[:,j-1] + gather[:,j+1])

    
    
    
    
    
    
    

