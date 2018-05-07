import pandas as pd
import os
import math
import random
    
def gen_set(train_path):
    """
    params: train_path: String containing path to metadata file. Expects columns 'label','fname','manually_verified'.
            plot_dist: Boolean value specifying whether a plot of distribution of manually verified labels is to be generated.
    return: train_set: dictionary with labels as keys, each containing a list of audio file names representing the train set for that label.
            test_set:  same format as train_set representing the test set.
            train_stats: dictionary with labels as keys, each containing a dictionary of keys 0 or 1 representing the 'manually_verified' column of the metadata.
                         Contains the number of samples in the train set for each.
            test_stats: Same as train_stats but for the test set.
    """
    if not isinstance(train_path, str):
        raise TypeError("Input to gen_set needs to be a string specifying the path to the metadata file.")
    df = pd.read_csv(train_path)
    mv_label = [0,1]
    
    ratio = 0.8
    train_set = dict()
    test_set = dict()
    train_stats = dict()
    test_stats = dict()
    for l in mv_label:
        df_meta = df.loc[df['manually_verified']==l]
    
        label_dict = df_meta.groupby('label')['fname'].apply(lambda x: x.tolist())
        label_dict = label_dict.to_dict()
        all_labels = label_dict.keys()
        for label in all_labels:
            if label not in train_set:
                train_set[label] = []
                test_set[label] = []
                train_stats[label] = dict()
                test_stats[label] = dict()
    
            random.Random(10).shuffle(label_dict[label])
            point = int(len(label_dict[label])*ratio)
            train_set[label] += label_dict[label][0:point]
            test_set[label] += label_dict[label][point:]
            train_stats[label][l] = len(label_dict[label][0:point])
            test_stats[label][l] = len(label_dict[label][point:])
    

    return train_set, test_set, train_stats, test_stats
