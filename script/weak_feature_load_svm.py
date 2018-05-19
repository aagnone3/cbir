import glob
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import os
import pdb

#from weak_feature_extractor import *
from weak_feature_extractor import feat_extractor as fe

audio_folder = '/Users/avidwans/Documents/CBIR/CBIR_data/audio_train/'
#list_files = glob.glob(audio_folder + '*.wav')

split_files = '../dcase2018_gen/data_splits/main_baseline/'
train_lbl_files = split_files + 'train.meta'
train_meta = split_files + 'train.meta'
test_lbl_files = split_files + 'test.meta'
test_meta = split_files + 'test.meta'

# read train and test labels from file
#tr_lbl_onehot = pd.read_csv(train_lbl_file, skiprows=[0], header=None)
#ts_lbl_onehot = pd.read_csv(test_lbl_file, skiprows=[0], header=None)

with open(train_lbl_files) as f:
    ncols = len(f.readline().split(','))

#print(ncols)
tr_lbl_onehot = np.loadtxt(train_lbl_files, delimiter=',', skiprows=1, usecols=range(2,3))
ts_lbl_onehot = np.loadtxt(test_lbl_files, delimiter=',', skiprows=1, usecols=range(2,3))

#convert one hot labels to integer
#tr_lbl = [ np.where(r==1)[0][0] for r in tr_lbl_onehot ]
#print(len(tr_lbl))
#print(tr_lbl[0:5])
#ts_lbl = [ np.where(r==1)[0][0] for r in ts_lbl_onehot ]

tr_lbl = tr_lbl_onehot
ts_lbl = ts_lbl_onehot

tr_files = pd.read_csv(train_lbl_files)
tr_features = np.empty([len(tr_lbl), 1024])
ts_files = pd.read_csv(test_lbl_files)
ts_features = np.empty([len(ts_lbl), 1024])

tr_features = np.loadtxt("train_features.csv", delimiter=',')
ts_features = np.loadtxt("test_features.csv", delimiter=',')


linsvm_model = svm.SVC(kernel = 'linear', probability=True).fit(tr_features,tr_lbl)
ts_pred = linsvm_model.predict_proba(ts_features)

chk = sorted(zip(linsvm_model.classes_,ts_pred[0]), key=lambda x:x[1])[-3:]

import pdb; pdb.set_trace()
thefile = open('test_pred_prob_class.txt','w')
for item in chk:
        thefile.write("%s\n" % item)
thefile.close()

accuracy = linsvm_model.score(ts_features, ts_lbl)
print(accuracy)


