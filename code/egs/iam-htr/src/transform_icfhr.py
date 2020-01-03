#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
from shutil import copyfile
import sys

# Transform ICFHR texts 

if __name__ == '__main__':

    # Subfolder such as '30887' for Bentham and '30883' for Goethe or put "all" for all subfolders to be treated
    subfolder = sys.argv[1]
        
    # Create one train/val/split as for IAM 

    data_folder = os.path.join(os.getcwd(),'data_icfhr/general_data')
    if subfolder != 'all':
        dirs = [subfolder]
    else:
        dirs = os.listdir(data_folder)

    # All lines
    lines_old = {}

    for dir in dirs:
        path = os.path.join(data_folder,dir)
        dirs2 = os.listdir(path)
        for dir2 in dirs2:
            path2 = os.path.join(path,dir2)
            files = os.listdir(path2)
            for file in files:
                if "txt" in file:
                    img_file = file[:-8]
                    filename = os.path.join(path2,file)
                    with open(filename,'r') as f:
                        lines = f.readlines()[0]
                        lines_old[img_file] = lines


    # Replace delimiter as in IAM training
    lines_dict = {}
    for k,v in lines_old.items():
        lines_dict[k] = ' '.join(v.replace(' ','@'))

    lines = []
    for k,v in lines_dict.items():
        lines.append(k+" "+v+"\n")

    folder_imgs = os.path.join(os.getcwd(),'data_icfhr/original/lines')
    if not os.path.exists(folder_imgs):
        os.makedirs(folder_imgs)


    for dir in dirs:
        path = os.path.join(data_folder,dir)
        dirs2 = os.listdir(path)
        for dir2 in dirs2:
            path2 = os.path.join(path,dir2)
            files = os.listdir(path2)
            for file in files:
                if "txt" in file:
                    img_file = file[:-4]
                    img_path = os.path.join(path2,img_file)
                    dst_path = os.path.join(folder_imgs,img_file)
                    copyfile(img_path,dst_path)

    folder_txt = os.path.join(os.getcwd(),'data_icfhr/lang/puigcerver/lines/char')
    if not os.path.exists(folder_txt):
        os.makedirs(folder_txt)

    txt_filename = os.path.join(folder_txt,'original.txt')
    with open(txt_filename,'w') as f:
        f.writelines(lines)