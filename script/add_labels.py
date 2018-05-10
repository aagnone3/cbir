#!/usr/bin/env python2
import pickle
import pandas as pd

BINARIZER_FN = "binarizer.pkl"
ENCODER_FN = "encoder.pkl"

# load in the label encoder and label binarizer
with open(BINARIZER_FN, 'rb') as fp:
    binarizer = pickle.load(fp)
with open(ENCODER_FN, 'rb') as fp:
    encoder = pickle.load(fp)


def add_labels(df):
    # add the encoded labels
    df['label'] = encoder.transform(df['description'])

    # add the binarized labels
    binarized_labels = binarizer.transform(df['description'])
    bdf = pd.DataFrame(binarized_labels, columns=binarizer.classes_)

    # concatenate the meta data with the binarized labels
    return pd.concat((df, bdf), axis=1)


# read in the files
train_df = pd.read_csv('train.meta')
test_df = pd.read_csv('test.meta')

# add the labels
train_df = add_labels(train_df)
test_df = add_labels(test_df)

# save the files
import pdb; pdb.set_trace()
train_df.to_csv('train.full.meta', index=False)
test_df.to_csv('test.full.meta', index=False)
