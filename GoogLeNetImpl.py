from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Ftrl, SGD, Adadelta
from tensorflow.python.client import device_lib
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import concatenate

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
    print(e)

tf.config.set_soft_device_placement(True)

#██████╗░██████╗░███████╗██████╗░██████╗░░█████╗░░█████╗░███████╗░██████╗░██████╗
#██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝
#██████╔╝██████╔╝█████╗░░██████╔╝██████╔╝██║░░██║██║░░╚═╝█████╗░░╚█████╗░╚█████╗░
#██╔═══╝░██╔══██╗██╔══╝░░██╔═══╝░██╔══██╗██║░░██║██║░░██╗██╔══╝░░░╚═══██╗░╚═══██╗
#██║░░░░░██║░░██║███████╗██║░░░░░██║░░██║╚█████╔╝╚█████╔╝███████╗██████╔╝██████╔╝
#╚═╝░░░░░╚═╝░░╚═╝╚══════╝╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚════╝░╚══════╝╚═════╝░╚═════╝░
start = time.perf_counter()

# Importing the preprocessed data for training and

xy_train_df = pd.read_csv('xy_train.csv')
x_test_df = pd.read_csv('x_test.csv')
# preprocess image data


def load_image(file):
    try:
        image = Image.open(
            file
        ).convert('RGB').resize((270, 270))
        arr = np.array(image)
    except:
        arr = np.zeros((270, 270, 3))
    return arr


# loading images:
x_image = np.array([load_image(i) for i in tqdm(xy_train_df.image)])


plt.imshow(x_image[0, :, :, 0])
#plt.show()

# labels:
y_cancer = xy_train_df.is_cancer
y_type = xy_train_df.type.astype('category').cat.codes

len_cancer = len(y_cancer.unique())
len_type = len(y_type.unique())
print('unique values for cancer category', len_cancer, y_cancer.unique())
print('unique values for type category', len_type, y_type.unique())

# splitting:

x_tr_image, x_vl_image, y_tr_cancer, y_vl_cancer, y_tr_type, y_vl_type = train_test_split(
    x_image,
    y_cancer,
    y_type,
    test_size=0.2)

print(np.shape(x_tr_image))
print(np.shape(x_vl_image))
print(np.shape(y_tr_cancer))
print(np.shape(y_vl_cancer))
print(np.shape(y_tr_type))
print(np.shape(y_vl_type))

# preprocess text data

vocab_size = 40000
max_len = 50


# ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░
# ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░
# ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░
# ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░
# ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗
# ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝

in_image = keras.Input(batch_shape=(None, 270, 270, 3))

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
 
  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer

def GoogLeNet(in_image):

  # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
  X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(in_image)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # convolutional layer: filters = 64, strides = 1
  X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)

  # convolutional layer: filters = 192, kernel_size = (3,3)
  X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 1st Inception block
  X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)

  # 2nd Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 3rd Inception block
  X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)

  # Extra network 1:
  X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
  X1 = Flatten()(X1)
  X1 = Dense(1024, activation = 'relu')(X1)
  X1 = Dropout(0.7)(X1)
  X1 = Dense(5, activation = 'softmax')(X1)

  
  # 4th Inception block
  X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 5th Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 6th Inception block
  X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)

  # Extra network 2:
  X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
  X2 = Flatten()(X2)
  X2 = Dense(1024, activation = 'relu')(X2)
  X2 = Dropout(0.7)(X2)
  X2 = Dense(1000, activation = 'relu')(X2)
  
  
  # 7th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, 
                      f3_conv5 = 128, f4 = 128)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # 8th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

  # 9th Inception block
  X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)

  # Global Average pooling layer 
  X = GlobalAveragePooling2D(name = 'GAPL')(X)

  # Dropoutlayer 
  X = Dropout(0.4)(X)

  # output layer 
  X = Dense(1000, activation = 'relu')(X)
  
  # model
  #model = Model(in_image, [X, X1, X2], name = 'GoogLeNet')

  p_cancer = Dense(len_cancer, activation='softmax', name='cancer')(X)
  p_type = Dense(len_type, activation='softmax', name='type')(X)

  model = tf.keras.models.Model(
    inputs={
        'image': in_image
    },
    outputs={
        'cancer': p_cancer,
        'type': p_type,
    },
  )
 
  return model


model = GoogLeNet(in_image)

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss={
        'cancer': 'sparse_categorical_crossentropy',
        'type': 'sparse_categorical_crossentropy',
    },
    loss_weights={
        'cancer': 0.4,
        'type': 0.5,
    },
    metrics={
        'cancer': ['SparseCategoricalAccuracy'],
        'type': ['SparseCategoricalAccuracy'],
    },
)


#████████╗██████╗░░█████╗░██╗███╗░░██╗██╗███╗░░██╗░██████╗░
#╚══██╔══╝██╔══██╗██╔══██╗██║████╗░██║██║████╗░██║██╔════╝░
#░░░██║░░░██████╔╝███████║██║██╔██╗██║██║██╔██╗██║██║░░██╗░
#░░░██║░░░██╔══██╗██╔══██║██║██║╚████║██║██║╚████║██║░░╚██╗
#░░░██║░░░██║░░██║██║░░██║██║██║░╚███║██║██║░╚███║╚██████╔╝
#░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝╚═╝╚═╝░░╚══╝░╚═════╝░
epochs = 1
batsize = 32
model.summary()
with tf.device('/GPU:0'):

  history = model.fit(
    x={
        'image': x_tr_image
    },
    y={
        'cancer': y_tr_cancer,
        'type': y_tr_type,
    },
    epochs=epochs,
    batch_size=batsize,
    validation_data=(
        {
            'image': x_vl_image
        },
        {
            'cancer': y_vl_cancer,
            'type': y_vl_type,
        }),
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_cancer_loss', patience=epochs)
    ],
    verbose=1
  )

elapsed = time.perf_counter() - start

print('Elapsed %.3f mins.' % (elapsed/60))
t = time.localtime()
timestamp = time.strftime('%d-%b-%Y_%H-%M-%S', t)
tist = timestamp

#Plotting the Network Architecture

model.save('models/VRNN_%d_epochs_%s.h5' % (epochs, tist))

x_test_image = np.array([load_image(i) for i in tqdm(x_test_df.image)])

y_predict = model.predict(
    {
        'image': x_test_image
    }
)

iscancer_predicted = y_predict['cancer']
print(iscancer_predicted)
iscancer_category_predicted = np.argmax(iscancer_predicted, axis=1)
print(iscancer_category_predicted)

type_predicted = y_predict['type']
type_category_predicted  = np.argmax(type_predicted, axis=1)

pd.DataFrame(
    {'id': x_test_df.id, 'type': type_category_predicted,
     'cancer': iscancer_category_predicted}).to_csv('sample_submission.csv', index=False)
print("history Callbacks:  ", history.History)

#██████╗░██╗░░░░░░█████╗░████████╗
#██╔══██╗██║░░░░░██╔══██╗╚══██╔══╝
#██████╔╝██║░░░░░██║░░██║░░░██║░░░
#██╔═══╝░██║░░░░░██║░░██║░░░██║░░░
#██║░░░░░███████╗╚█████╔╝░░░██║░░░
#╚═╝░░░░░╚══════╝░╚════╝░░░░╚═╝░░░

##Visualizing training and Validation
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(20, 15))
plt.plot(history.history['cancer_sparse_categorical_accuracy'])
plt.plot(history.history['val_cancer_sparse_categorical_accuracy'])
plt.title('Model cancer accuracy', fontsize=28)
plt.ylabel('accuracy', fontsize=28)
plt.xlabel('epoch', fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left', fontsize=18)
plt.grid(color='k', linewidth=0.2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.show()
plt.savefig('Graphs/foo_%d_epochs_%s.png' % (epochs, tist))

# summarize history for accuracy
plt.figure(figsize=(20, 15))
plt.plot(history.history['type_sparse_categorical_accuracy'])
plt.plot(history.history['val_type_sparse_categorical_accuracy'])
plt.title('Model type accuracy', fontsize=28)
plt.ylabel('accuracy', fontsize=28)
plt.xlabel('epoch', fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left', fontsize=18)
plt.grid(color='k', linewidth=0.2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.show()
plt.savefig('Graphs/foo1_%d_epochs_%s.png' % (epochs, tist))

# summarize history for loss
plt.figure(figsize=(20, 15))
plt.plot(history.history['cancer_loss'])
plt.plot(history.history['val_cancer_loss'])
plt.title('Model cancer loss', fontsize=28)
plt.ylabel('loss', fontsize=28)
plt.xlabel('epoch', fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(color='k', linewidth=0.2)
#plt.show()
plt.savefig('Graphs/foo2_%d_epochs_%s.png' % (epochs, tist))

plt.figure(figsize=(20, 15))
plt.plot(history.history['type_loss'])
plt.plot(history.history['val_type_loss'])
plt.title('Model type loss', fontsize=28)
plt.ylabel('loss', fontsize=28)
plt.xlabel('epoch', fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(color='k', linewidth=0.2)
#plt.show()
plt.savefig('Graphs/foo3_%d_epochs_%s.png' % (epochs, tist))

lst1 = history.history['cancer_sparse_categorical_accuracy']
lst2 = history.history['val_cancer_sparse_categorical_accuracy']
lst3 = history.history['type_sparse_categorical_accuracy']
lst4 = history.history['val_type_sparse_categorical_accuracy']
lst5 = history.history['cancer_loss']
lst6 = history.history['val_cancer_loss']
lst7 = history.history['type_loss']
lst8 = history.history['val_type_loss']

df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8)), columns= ['cancer_sparse_categorical_accuracy',
'val_cancer_sparse_categorical_accuracy', 'type_sparse_categorical_accuracy', 'val_type_sparse_categorical_accuracy',
'cancer_loss', 'val_cancer_loss', 'type_loss', 'val_type_loss'])
df.to_csv('Resnetmodel_Batch_%d_epochs%d__%s.csv'%(batsize,epochs, tist))