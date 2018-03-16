# coding: utf-8
"""
Current usage:
Evalutes the results of an EventDetector output, which is various files with a filename like *results*.txt

Workflow:
<dir> has the results files
find <dir> -type f -name "*results*" > files.lst
python esc50_benchmark.py -i files.lst
rm files.lst
Open output files for further analysis.

"""
import numpy as np
from os import path
from functools import partial
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import Pool
from glob import glob
from collections import defaultdict

ESC50_META_FN = "/home/aagnone/projects/audio_event_search/exp1/esc50.csv"
META = pd.read_csv(ESC50_META_FN)
SOUND_TYPES = {
    "animal": ("dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"),
    "natural": ("rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm"),
    "nonspeech": ('crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping'),
    "interior": ('door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking'),
    "urban": ('helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw')
}


def full_fn_to_ref_fn(full_fn):
    return full_fn.split('-')[1]


def truncate_hit_id(hit_id):
    return hit_id[-2]


def truncate_probe_fn(fn):
    return path.splitext(path.basename(fn))[0].split('-')[1]


def load_res_file(type_to_class, fn_to_class, fn):
    df = pd.read_csv(fn, sep=' ', names=['ref_fn', 'hit_id', 'probe_fn', 't1', 't2'])
    base_fn = path.basename(fn)
    df['filename'] = base_fn[:base_fn.index("_search")].split('_')[0]
    df['ref_fn'] = df['filename'].map(full_fn_to_ref_fn).astype(int)
    df['hit_id'] = df['hit_id'].map(truncate_hit_id)
    df['probe_fn'] = df['probe_fn'].map(truncate_probe_fn).astype(int)
    df['ref_class'] = df['ref_fn'].map(fn_to_class.get)
    df['probe_class'] = df['probe_fn'].map(fn_to_class.get)
    df['ref_cat'] = df['ref_class'].map(type_to_class.get)
    df['probe_cat'] = df['probe_class'].map(type_to_class.get)
    return df


def parse_results(results_fns):
    KEEP_COLS = ('ref_fn', 'probe_fn', 'ref_class', 'probe_class', 'ref_cat', 'probe_cat',
            'hit_id', 't1', 't2', 'filename', 'fold', 'take')

    fn_to_class = {
        fn: META.loc[META['src_file'] == fn, 'category'].iloc[0]
        for fn in META['src_file'].unique()
    }

    type_to_class = {
        sound_type: sound_class
        for sound_class, sound_types in SOUND_TYPES.iteritems()
        for sound_type in sound_types
    }

    df = pd.concat(Pool(16).map(partial(load_res_file, type_to_class, fn_to_class), results_fns))
    print "asdfasdfasdf"
    df2 = df.merge(META, on="filename")
    return df2.loc[:, KEEP_COLS]


def precision(df, group="class", top_k=5):
    counts = defaultdict(lambda: defaultdict(int))
    ref = "ref_{}".format(group)
    probe = "probe_{}".format(group)
    for query, grp in df.groupby(ref):
        y_pred = df[probe] == query
        y_true = df[ref] == query
        tp = np.sum(y_pred & y_true)
        fp = np.sum(y_pred & ~y_true)
        tn = np.sum(~y_pred & ~y_true)
        fn = np.sum(~y_pred & y_true)
        counts[query]['tp'] = tp
        counts[query]['fp'] = fp
        counts[query]['fn'] = fn
        counts[query]['tn'] = tn

    stats = pd.DataFrame(counts).transpose()
    stats['Recall'] = 100.0 * stats['tp'] / (stats['tp'] + stats['fn'])
    stats['Precision'] = 100.0 * stats['tp'] / (stats['tp'] + stats['fp'])
    return stats.sort_values('Recall', ascending=False)


def analyze_results(df):
    fn1 = "stats_sound_type.csv"
    fn2 = "stats_sound_classes.csv"
    precision(df, group="class").to_csv(fn1, index=False)
    precision(df, group="cat").to_csv(fn2, index=False)
    print "Basic stats are available at {} and {}".format(fn1, fn2)


def main(results_fn):
    print "Reading in list of results file names."
    results_fns = pd.read_csv(results_fn, header=None)[0].values

    print "Parsing results for {} files.".format(len(results_fns))
    df = parse_results(results_fns)

    print "Computing stats on the parsed results."
    analyze_results(df)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", dest="results_fn", help="Path to file with paths to results files (no header).")
    return parser.parse_args()


def cli():
    if __name__ == '__main__':
        args = parse_args()
        main(args.results_fn)
cli()
