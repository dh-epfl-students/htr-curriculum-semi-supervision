#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
import sys

# Create sampled datasets 
if __name__ == '__main__':
    folder = sys.argv[1]

    if folder != 'data_iam' and folder != 'data_washington' and folder != 'data_icfhr':
        raise Exception("not a valid argument")
    data_folder = os.path.join(os.getcwd(),folder)

    # Original files
    path_folder = data_folder + '/lang/puigcerver/lines/char/'
    filename_tr = 'tr.txt'
    tr_path = os.path.join(path_folder,filename_tr)
    filename_val = 'va.txt'
    val_path = os.path.join(path_folder,filename_val)

    # All lines
    lines = []

    # Read original train txt file
    with open(tr_path,'r') as f:
        lines = f.readlines()
    tr_len = len(lines)
    
    # Read original val txt file
    with open(val_path,'r') as f:
        lines.extend(f.readlines())
        
    val_len = len(lines)-tr_len
    total = len(lines)

    random.shuffle(lines)

    # CAREFUL: These lines are to be changed depending on your usage
    print(len(lines))
    if len(lines)>=460:
        tr_sampled = lines[:200]
        tr_unlabelled = lines[200:500]
        val_sampled = lines[500:585]
        
    else:
        print(len(lines))
        tr_sampled = lines[:150]
        tr_unlabelled = lines[150:300]
        val_sampled = lines[300:]

    print(len(lines))

    filename_tr = 'tr_sampled.txt'
    tr_path = os.path.join(path_folder,filename_tr)
    filename_tr_unlabel = 'tr_unlabelled_semi_supervised_sampled.txt'
    tr_unlabel_path = os.path.join(path_folder,filename_tr_unlabel)
        
    filename_val = 'va_sampled.txt'
    val_path = os.path.join(path_folder,filename_val)

    with open(tr_path,'w') as f:
        f.writelines(tr_sampled)

    with open(val_path,'w') as f:
        f.writelines(val_sampled)

    with open(tr_unlabel_path,'w') as f:
        f.writelines(tr_unlabelled)