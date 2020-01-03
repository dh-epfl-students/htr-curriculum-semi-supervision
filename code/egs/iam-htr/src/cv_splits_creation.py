#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
import sys
#Create cross-validation splits

if __name__ == '__main__':

    folder = sys.argv[1]
    reduced = sys.argv[2]

    if folder != 'data_iam' and folder != 'data_washington' and folder != 'data_icfhr':
        raise Exception("not a valid argument")

    data_folder = os.path.join(os.getcwd(),folder)
    #original files
    path_folder = data_folder + '/lang/puigcerver/lines/char/'
    
    if reduced == 'True':
        filename_tr = 'tr_sampled.txt'
        filename_val = 'va_sampled.txt'
    else:
        filename_tr = 'tr.txt'
        filename_val = 'va.txt'

    tr_path = os.path.join(path_folder,filename_tr)
    
    val_path = os.path.join(path_folder,filename_val)

    #All lines
    lines = []

    #Read original train txt file
    with open(tr_path,'r') as f:
        lines = f.readlines()
    tr_len = len(lines)

    #Read original val txt file
    with open(val_path,'r') as f:
        lines.extend(f.readlines())
        
    val_len = len(lines)-tr_len
    total = len(lines)

    random.shuffle(lines)
    #number of folds of the cross validation in order to keep the default network parameters
    k_folds = math.ceil((val_len+tr_len)/val_len)

    #to store the cross-validation splits
    cv_tuples = {}

    #sliding window of a validation length
    for i in range(k_folds):
        val = lines[i*val_len:(i+1)*val_len]
        train = lines[:i*val_len] + lines[(i+1)*val_len:]
        cv_tuples[i] = (train,val)
        
    #The last split has less data as we have taken the ceil of the ratio
    #it must be balanced with its respective train split
    diff = val_len - len(cv_tuples[k_folds-1][1])
    rest = cv_tuples[k_folds-1][0][-diff:]
    new_tuple = (cv_tuples[k_folds-1][0][:-diff],cv_tuples[k_folds-1][1] + rest)
    cv_tuples[k_folds-1] = new_tuple


    #Make Directories to save CV splits
    path_folder_cv = data_folder + '/lang/puigcerver/lines/char/cross_validation/'
    if not os.path.exists(path_folder_cv):
        os.mkdir(path_folder_cv)
    for i in range(0,k_folds):
        path = path_folder_cv + 'cv'+str(i)
        if not os.path.exists(path):
            os.mkdir(path)
            
            
    #Saving CV splits
    def save_cv_split(split_train,split_val,number):
        filename_tr = path_folder_cv + 'cv' + str(number) + '/tr.txt'
        filename_va = path_folder_cv + 'cv' + str(number) + '/va.txt'
        
        with open(filename_tr,'w') as f:
            f.writelines(split_train)
        
        with open(filename_va,'w') as f:
            f.writelines(split_val)
            

    for i in range(k_folds):
        split = cv_tuples[i]
        save_cv_split(split[0],split[1],i)


    print("Total",total)
