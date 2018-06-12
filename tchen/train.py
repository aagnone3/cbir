import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D, Reshape,
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU)
from keras.initializers import lecun_normal
import numpy as np
from evaluation_measures import get_f_measure_by_class
from data_generator import DataGenerator, AudioGenerator
import pandas as pd
from keras import metrics
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
print (K.floatx())
print (K.epsilon())
print (K.image_dim_ordering())
print (K.image_data_format())
print (K.backend())

from train_utils import base_model_1, benchmark_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

mlb = MultiLabelBinarizer()
import pandas as pd


classes_num = 41
dropout_rate = 0.25
batch_size = 64
audio_path = 'audio/audio_train/'
mode = 1

df = pd.read_csv('metadata/train_set.csv')
df['fname'] = audio_path + df['fname']
df_test = pd.read_csv('metadata/test_set.csv')
df_test['fname'] = audio_path + df_test['fname']
X_train_fn, y_train = df.fname.values, df.label.values
X_test_fn, y_test = df_test.fname.values, df_test.label.values

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
enc = OneHotEncoder(sparse=False)
y_train_int = label_enc.fit_transform(y_train)
y_train_int = y_train_int.reshape(len(y_train_int), 1)
y_train_one_hot = enc.fit_transform(y_train_int)

y_test_int = label_enc.transform(y_test)
y_test_int = y_test_int.reshape(len(y_test_int), 1)
y_test_one_hot = enc.transform(y_test_int)


# Create audio generator
audio_gen = AudioGenerator(batch_size=batch_size, fns=X_train_fn, labels=y_train_one_hot, mode=mode)
valid_gen = AudioGenerator(batch_size=batch_size, fns=X_test_fn, labels=y_test_one_hot, mode=mode)
l, Sxx = audio_gen.rnd_one_sample()

num_train = audio_gen.get_train_test_num()
num_test = valid_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size
image_shape = Sxx.shape
print(image_shape)


# Attention CNN
model, model_name = base_model_1(image_shape, classes_num, dropout_rate)
print(model.summary())


# opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
model.compile(optimizer='Adam', loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch, epochs=100,
                    validation_data=valid_gen.next_train(), validation_steps=validation_step)
# model.save('models/attention_base_model_1_delta.h5')
# model = load_model('models/attention_base_model_1_delta.h5')
X_test, y_true = audio_gen.get_test()

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_true, y_pred))

# with open('results/f1_score_2018_task_2.txt', 'ab') as fw:
#     fw.write('model name: %s mode: %d f1: %f\n' % (model_name, mode, macro_f_measure))

