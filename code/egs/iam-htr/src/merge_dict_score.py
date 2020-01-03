import pickle
import os
import sys

# Merge dictionary from the cross validation scoring to have one validation score per sample

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {}
    
    for key, value in dict2.items():
        dict3[key] = value
        
    for key, value in dict1.items():
        dict3[key] = value   

    return dict3


if __name__ == '__main__':

    folder = sys.argv[1]

    if folder != 'transfer' and folder != 'bootstrap':
        raise Exception("not a valid argument")

    merged_file = "exper/puigcerver17/train/dict_"+folder+"_final.pkl"
    dict_scores_folder = 'exper/puigcerver17/train/'+folder+'/scores'
    dirs = os.listdir(dict_scores_folder)
    dict_all = {}
    
    for file in dirs:
        with open(os.path.join(dict_scores_folder,file),'rb') as f:
            x = pickle.load(f)
            y = mergeDict(x,dict_all)
            dict_all = y
    
    print(len(list(dict_all.keys())))
    
    with open(merged_file,'wb') as f:
        pickle.dump(dict_all,f)