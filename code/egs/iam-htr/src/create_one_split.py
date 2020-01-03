#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
import sys

# Create one train/val/split as for IAM 
if __name__ == '__main__':

    folder = sys.argv[1]

    if folder != 'data_icfhr' and folder != 'data_washington':
        raise Exception("not a valid argument")

    data_folder = os.path.join(os.getcwd(),folder)
    # Original files
    path_folder = data_folder + '/lang/puigcerver/lines/char/'
    filename = 'original.txt'
    path = os.path.join(path_folder,filename)

    # All lines
    lines_old = []

    # Read original train txt file
    with open(path,'r') as f:
        lines_old = f.readlines()

    if folder=='data_washington':
        # Replace delimiter as in IAM training
        lines = []
        for x in lines_old:
            lines.append(x.replace('<sp>','@'))
    else:
        lines = lines_old

    # Retrieve the letters of IAM
    path_syms_ctc = os.path.join(os.getcwd(),'exper/puigcerver17/train/syms_ctc.txt')
    with open(path_syms_ctc,'r') as f:
        letters = f.readlines()
    different_letters = []
    for let in letters:
        different_letters.append(let.split(" ")[0])

    # Add space and line break
    different_letters.append(" ")
    different_letters.append("\n")

    # Retrieve all letters used in washington db
    all_letters = []
    for line in lines:
        for car in line:
            if car not in all_letters:
                all_letters.append(car)

    all_letters = set(all_letters)
    different_letters = set(different_letters)

     #Letters to be removed
    wrong_letters = list(all_letters - different_letters)
    print(wrong_letters)

    if folder=='data_washington':
        # Remove wrong letters
        lines_filtered = []
        for line in lines:
            for car in wrong_letters:
                line = line.replace(car+" ","")
            lines_filtered.append(line)

        lines = lines_filtered

    random.shuffle(lines)
    total = len(lines)

    # Splits and save
    tr_idx = math.ceil(0.6*total)
    va_idx = math.ceil(0.7*total)

    filename = 'tr.txt'
    path = os.path.join(path_folder,filename)
    split = lines[:tr_idx]
    with open(path,'w') as f:
        f.writelines(split)

    filename = 'va.txt'
    path = os.path.join(path_folder,filename)
    split = lines[tr_idx:va_idx]
    with open(path,'w') as f:
        f.writelines(split)

    filename = 'te.txt'
    path = os.path.join(path_folder,filename)
    split = lines[va_idx:]
    with open(path,'w') as f:
        f.writelines(split)