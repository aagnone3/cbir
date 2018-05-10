import pandas as pd
import numpy as np
import os
import math
import random
import pickle
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

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
    ver_label = [0,1]

    ratio = 0.8
    train_set = list()
    test_set = list()
    train_stats = dict()
    test_stats = dict()
    for l in ver_label:
        df_meta = df.loc[df['manually_verified']==l]

        label_dict = df_meta.groupby('label')['fname'].apply(lambda x: x.tolist())
        label_dict = label_dict.to_dict()
        all_labels = label_dict.keys()
        for label in all_labels:
            if label not in train_set:
                train_stats[label] = dict()
                test_stats[label] = dict()

            random.Random(10).shuffle(label_dict[label])
            point = int(len(label_dict[label])*ratio)
            train_set.extend(
                [
                    (label, fn)
                    for fn in label_dict[label][0:point]
                ]
            )
            test_set.extend(
                [
                    (label, fn)
                    for fn in label_dict[label][point:]
                ]
            )
            train_stats[label][l] = len(label_dict[label][0:point])
            test_stats[label][l] = len(label_dict[label][point:])

    train_df = pd.DataFrame(train_set, columns=['description', 'fname'])
    test_df = pd.DataFrame(test_set, columns=['description', 'fname'])

    binarizer = LabelBinarizer().fit(train_df["description"])
    encoder = LabelEncoder().fit(train_df["description"])

    train_df["label"] = encoder.transform(train_df["description"])
    test_df["label"] = encoder.transform(test_df["description"])

    return train_df, test_df, binarizer, encoder


train, test, binar, encod = gen_set("train.csv")

with open("binarizer.pkl", 'wb') as fp:
    pickle.dump(binar, fp)
with open("encoder.pkl", 'wb') as fp:
    pickle.dump(encod, fp)
train.to_csv("train.meta", index=False)
test.to_csv("test.meta", index=False)
