#/usr/bin/env python2
from __future__ import print_function
import os
import yaml
from os import path
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_curve, accuracy_score
from util import mapk

from argparse import ArgumentParser


BENCHMARK_FN = "benchmark.yaml"


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(dest="labels_fn",
                        help="Path to file with a 'label' column with the true class index. ")
    parser.add_argument(dest="preds_fn",
                        help="Path to file with a 'label' column with the 3 highest probability class indices. "
                             "The class indices may be acquired via the class encoder.")
    parser.add_argument(dest="tag", help="Tag to describe the model/features of the current result.")
    parser.add_argument("--update", dest="update", required=False, default=False)
    return parser


def load_data(labels_fn, preds_fn):
    y_true = pd.read_csv(labels_fn)['label'].astype(int).values.reshape((-1, 1))
    y_pred = pd.read_csv(preds_fn)['label'].apply(lambda d: np.array(d.split(' '), dtype=int))
    return y_true, y_pred


args = build_parser().parse_args()
y_true, y_pred_3,  = load_data(args.labels_fn, args.preds_fn)
y_pred = list(map(lambda x: x[0], y_pred_3))

# MAP @ 3
map3 = 100.0 * mapk(y_true, y_pred_3, 3)
print("MAP @ 3: {:.2f}".format(map3))

# Raw accuracy
accuracy = 100.0 * accuracy_score(y_true, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

if args.update:
    print("Updating the benchmark.")

    # load the current benchmark results
    with open(BENCHMARK_FN) as fp:
        benchmark = yaml.load(fp) or {}

    # add the current result
    benchmark[args.tag] = {
        "map@3":  "{:.3f}".format(map3),
        "accuracy": "{:.3f}".format(accuracy)
    }

    # persist the changes to disk
    with open(BENCHMARK_FN, 'w') as fp:
        yaml.dump(benchmark, fp)
else:
    print("Benchmark not updated. Use --update to update the benchmark.")
