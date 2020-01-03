import os
import math
import random
import sys

# Create sampled split for the baseline of Curriculum Learning - baseline-30%
if __name__ == '__main__':

    folder = sys.argv[1]
    percent = float(sys.argv[2])
    reduced = sys.argv[3]

    if folder != 'data_icfhr' and folder != 'data_washington':
        raise Exception("not a valid argument")

    data_folder = os.path.join(os.getcwd(),folder)
    # Original files
    path_folder = data_folder + '/lang/puigcerver/lines/char/'
    if reduced == 'True':
        filename = 'tr_sampled.txt'
    else:
        filename = 'tr.txt'
    path = os.path.join(path_folder,filename)

    # All lines
    lines = []

    # Read original train txt file
    with open(path,'r') as f:
        lines = f.readlines()

    sampled = random.sample(lines,int(percent*len(lines)))

    filename = 'tr_reduced.txt'
    path = os.path.join(path_folder,filename)
    with open(path,'w') as f:
        f.writelines(sampled)
    print(len(sampled))