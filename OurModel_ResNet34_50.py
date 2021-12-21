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
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout, Activation, Add
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Ftrl, SGD, Adadelta
import keras



# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

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
        ).convert("RGB").resize((270, 270))
        arr = np.array(image)
    except:
        arr = np.zeros((270, 270, 3))
    return arr


# loading images:
x_image = np.array([load_image(i) for i in tqdm(xy_train_df.image)])


plt.imshow(x_image[0])
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


# ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░
# ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░
# ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░
# ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░
# ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗
# ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝

#ResNet-50
def res_identity(x, filters):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x  # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(tf.keras.activations.relu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(tf.keras.activations.relu)(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(tf.keras.activations.relu)(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation(tf.keras.activations.relu)(x)

  return x


def res_conv(x, s, filters):
  '''
  here the input size changes'''
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(tf.keras.activations.relu)(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(tf.keras.activations.relu)(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                  kernel_regularizer=tf.keras.regularizers.L2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add
  x = Add()([x, x_skip])
  x = Activation(tf.keras.activations.relu)(x)

  return x


def resnet50(shape=(270, 270, 3)):

    input_im = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    #2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(1000, activation='softmax', kernel_initializer='he_normal')(x)

    #multi-class
    p_cancer = Dense(len_cancer, activation='softmax', name='cancer')(x)
    p_type = Dense(len_type, activation='softmax', name='type')(x)
    # define the model

    model = tf.keras.models.Model(inputs={'image': input_im},
                                  outputs={'cancer': p_cancer, 'type': p_type}, name="ResNet50")

    return model

#ResNet-34


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(
        filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet34(shape=(270, 270, 3)):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    p_cancer = Dense(len_cancer, activation='softmax', name='cancer')(x)
    p_type = Dense(len_type, activation='softmax', name='type')(x)
    model = tf.keras.models.Model(inputs={'image': x_input},
                                  outputs={'cancer': p_cancer, 'type': p_type}, name="ResNet34")
    return model


#Our Model
in_image = keras.Input(batch_shape=(None, 270, 270, 3))
cov = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(in_image)
pl = MaxPool2D((2, 2), strides=(2, 2))(cov)
pl = Conv2D(filters=64, kernel_size=(1, 1), strides=(
    1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=64, kernel_size=(3, 3), strides=(
    1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=128, kernel_size=(1, 1), strides=(
    1, 1), activation='relu', padding="same")(pl)
pl = BatchNormalization()(pl)
pl = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(pl)
pl = Conv2D(filters=256, kernel_size=(1, 1), strides=(
   1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=256, kernel_size=(3, 3), strides=(
   1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=256, kernel_size=(1, 1), strides=(
  1, 1), activation='relu', padding="same")(pl)
pl = BatchNormalization()(pl)
pl = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(pl)
pl = Conv2D(filters=256, kernel_size=(1, 1), strides=(
   1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=256, kernel_size=(3, 3), strides=(
   1, 1), activation='relu', padding="same")(pl)
pl = Conv2D(filters=256, kernel_size=(1, 1), strides=(
  1, 1), activation='relu', padding="same")(pl)
pl = BatchNormalization()(pl)
pl = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(pl)
flattened = Flatten()(pl)
fc1 = Dense(512)(flattened)
fc2 = Dense(512)(fc1)
#multi-objectives (each is a multi-class classification)
p_cancer = Dense(len_cancer, activation='softmax', name='cancer')(fc2)
p_type = Dense(len_type, activation='softmax', name='type')(fc2)
# model = keras.Model(
#     inputs={
#         'image': in_image
#     },
#     outputs={
#         'cancer': p_cancer,
#         'type': p_type,
#     },
# )

model = ResNet34()
modelname = format(model)
print('Model name:', modelname)
print
model.compile(
    optimizer=Nadam(learning_rate=0.0001),
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
batsize = 64
model.summary()
#with tf.device('/GPU:0'):

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
            monitor='val_type_loss', patience=epochs)
    ],
    verbose='auto'
)

elapsed = time.perf_counter() - start

print('Elapsed %.3f mins.' % (elapsed/60))
t = time.localtime()
timestamp = time.strftime('%d-%b-%Y_%H-%M-%S', t)
tist = timestamp

#Plotting the Network Architecture
tf.keras.utils.plot_model(
    model,
    to_file="Model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="LR",
    expand_nested=False,
    dpi=300,
)
model.save('models/VRNN_%d_epochs_%s.h5' % (epochs, tist))

#x_test_image = np.array([load_image(i) for i in tqdm(x_test_df.image)])


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


#saving model training & validation results
lst1 = history.history['cancer_sparse_categorical_accuracy']
lst2 = history.history['val_cancer_sparse_categorical_accuracy']
lst3 = history.history['type_sparse_categorical_accuracy']
lst4 = history.history['val_type_sparse_categorical_accuracy']
lst5 = history.history['cancer_loss']
lst6 = history.history['val_cancer_loss']
lst7 = history.history['type_loss']
lst8 = history.history['val_type_loss']

df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8)), columns=['cancer_sparse_categorical_accuracy',
                                                                                      'val_cancer_sparse_categorical_accuracy', 'type_sparse_categorical_accuracy', 'val_type_sparse_categorical_accuracy',
                                                                                      'cancer_loss', 'val_cancer_loss', 'type_loss', 'val_type_loss'])
df.to_csv('Resnet50model_Batch_%d_epochs%d__%s.csv' % (batsize, epochs, tist))

# y_predict = model.predict({'image': x_test_image})

# iscancer_predicted = y_predict['cancer']
# print(iscancer_predicted)
# iscancer_category_predicted = np.argmax(iscancer_predicted, axis=1)
# print(iscancer_category_predicted)

# type_predicted = y_predict['type']
# type_category_predicted  = np.argmax(type_predicted, axis=1)

# pd.DataFrame(
#     {'id': x_test_df.id, 'type': type_category_predicted,
#      'cancer': iscancer_category_predicted}).to_csv('sample_submission.csv', index=False)
