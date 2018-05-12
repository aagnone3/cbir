#!/usr/bin/env python
import numpy as np
import pickle
np.random.seed(1001)

import os
import yaml
from os import path
import shutil
import scipy

import librosa
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.cross_validation import StratifiedKFold

from keras import backend as K
from keras import losses, models, optimizers
from keras.activations import *
from keras.callbacks import *
from keras.layers import *
from keras.utils import Sequence, to_categorical

from util import audio_norm, Config, DataGenerator
from models import conv_2d, dummy_2d

TEST_RUN = False
WEIGHTS_DIR = "weights"
WAV_DIR = "/home/aagnone/data/dcase2018_gen"
PREDICTION_DIR = "predictions_2d_conv"
LOG_DIR = path.join("logs", PREDICTION_DIR)
BINARIZER_FN = "binarizer.pkl"
ENCODER_FN = "encoder.pkl"
DATA_DIR = "data"

if not os.path.exists(PREDICTION_DIR):
    os.mkdir(PREDICTION_DIR)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)


with open(BINARIZER_FN, 'rb') as fp:
    binarizer = pickle.load(fp)
with open(ENCODER_FN, 'rb') as fp:
    encoder = pickle.load(fp)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-tr', dest='train_fn')
    parser.add_argument('-te', dest='test_fn')
    parser.add_argument('--tag', dest='tag', help='Tag to apply when created output features.', default='')
    parser.add_argument('-c', dest='conf_fn', help='Path to conf file with training parameters.')
    return parser

def get_features(df, config, target_fn):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    if path.exists(target_fn):
        X = np.load(target_fn)["data"]
    else:
        for i, fname in tqdm(enumerate(df.index)):
            data, _ = librosa.core.load(fname, sr=config.sampling_rate, res_type="kaiser_fast")

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[i,] = data
        np.savez_compressed(target_fn, data=X)
    return X


def get_training_callbacks(fold_num, weights_fn):
    log_fn = path.join(PREDICTION_DIR, 'fold_%i' % i)
    checkpoint = ModelCheckpoint(weights_fn, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    tb = TensorBoard(log_dir=log_fn, write_graph=True)
    return [checkpoint, early, tb]


def make_submission_file(df, labels, probas, fn):
    top_3 = np.array(labels)[np.argsort(-probas, axis=1)[:, :3]]
    predicted_labels = [' '.join(map(str, list(x))) for x in top_3]
    df.index = df.index.map(path.basename)
    df['label'] = predicted_labels
    df[['label']].to_csv(fn)


def make_proba_file(df, labels, probas, fn):
    new_df = pd.DataFrame(probas, columns=binarizer.classes_)
    new_df.index = df.index.map(path.basename)
    new_df.to_csv(fn)


def get_data(train, test, config, tag):
    X_train = get_features(train, config, path.join(DATA_DIR, "train.{tag}.npz".format(tag=tag)))
    X_test = get_features(test, config, path.join(DATA_DIR, "test.{tag}.npz".format(tag=tag)))
    y_train = binarizer.transform(train.description.values.tolist())

    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return X_train, X_test, y_train


def setup(train_fn, test_fn, conf_fn, tag):
    train = pd.read_csv(train_fn).set_index("fname")
    test = pd.read_csv(test_fn).set_index("fname")
    class_names = train.columns[2:]
    LABELS = encoder.transform(class_names)

    train["label_idx"] = encoder.transform(train.description)
    train["description"] = train.description.map(str.encode)
    if TEST_RUN:
        train = train[:2000]
        test = test[:2000]

    if TEST_RUN:
        config = Config(sampling_rate=44100, audio_duration=2, n_folds=2,
                        max_epochs=1, use_mfcc=True, n_mfcc=40)
    else:
        with open(conf_fn) as fp:
            conf = yaml.load(fp)
        config = Config(**conf)

    X_train, X_test, y_train = get_data(train, test, config, tag)
    return X_train, X_test, y_train, config


def run(X_train, X_test, y_train, config):
    skf = StratifiedKFold(y_train.argmax(axis=1), n_folds=config.n_folds)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()

        # file names
        weights_fn = path.join(WEIGHTS_DIR, 'best_%d.h5' % i)
        train_preds_fn = PREDICTION_DIR + "/train_predictions_%d.npy" % i
        test_preds_fn = PREDICTION_DIR + "/test_predictions_%d.npy" % i
        submission_fn = PREDICTION_DIR + "/predictions_%d.csv" % i
        proba_fn = PREDICTION_DIR + "/probas_%d.csv" % i

        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
        print("#"*50)
        print("Fold: ", i)
        model = conv_2d(config)
        history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=get_training_callbacks(i, weights_fn),
                            batch_size=config.batch_size, epochs=config.max_epochs)
        model.load_weights(weights_fn)

        # Save train predictions
        train_predictions = model.predict(X_train, batch_size=config.batch_size, verbose=1)
        np.save(train_preds_fn, train_predictions)

        # Save test predictions
        test_probas = model.predict(X_test, batch_size=config.batch_size, verbose=1)
        np.save(test_preds_fn, test_probas)

        make_submission_file(test, LABELS, test_probas, submission_fn)
        make_proba_file(test, LABELS, test_probas, proba_fn)


def main():
    if __name__ == '__main__':
        args = build_parser().parse_args()
        run(setup(args))

main()
