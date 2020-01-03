import os
import math
import random
import sys

# Create new syms table for the new alphabet
# Add letters that are not in the IAM alphabet 
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
    lines = []

    # Read original train txt file
    with open(path,'r') as f:
        lines = f.readlines()

    # Retrieve all letters used in washington db
    all_letters = []
    for line in lines:
        for car in line:
            if car not in all_letters:
                all_letters.append(car)

    all_letters = set(all_letters)

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

    df = set(different_letters)

    # Letters to be removed
    wrong_letters = list(all_letters - df)

    different_letters.extend(wrong_letters)

    all_letters = different_letters

    print(all_letters)
    all_letters.remove("\n")
    all_letters.remove(" ")

    syms = []

    for letter,number in zip(all_letters,range(len(all_letters))):
        syms.append(str(letter) + " " + str(number) + "\n")
    print(syms)

    path_syms_ctc = os.path.join(os.getcwd(),'exper/puigcerver17/train/syms_ctc_icfhr.txt')
    with open(path_syms_ctc,'w') as f:
        f.writelines(syms)
