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

print(ncols)
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

#pdb.set_trace()
for index, row in tr_files.iterrows():
    #tr_features[index,:] = os.system('python /Users/avidwans/Documents/CBIR/weak_feature_extractor/feat_extractor.py '+ audio_folder + os.path.basename(row['fname']))
    tr_features[index,:] = fe.main(audio_folder + os.path.basename(row['fname']))

for index, row in ts_files.iterrows():
    ts_features[index,:] = fe.main(audio_folder + os.path.basename(row['fname']))

np.savetxt("train_features.csv",tr_features, delimiter=',')
np.savetxt("test_features.csv",ts_features, delimiter=',')

linsvm_model = svm.SVC(kernel = 'linear').fit(tr_features,tr_lbl)
ts_pred = linsvm_model.predict(ts_features)

accuracy = linsvm_model.score(ts_features, ts_lbl)
print(accuracy)

cm = confusion_matrix(ts_lbl, ts_pred)
print(cm)

thefile = open('test.txt','w')
for item in cm:
    thefile.write("%s\n" % item)
thefile.close()
