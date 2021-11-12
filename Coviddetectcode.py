from tqdm import tqdm
import PIL
from PIL import Image, ImageFilter
import pandas as pd
import time
import os,stat
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input,BatchNormalization,MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing


#Images Preprocessing
def load_data(folder):
    images = []
    for file in os.listdir(folder):
        file_id = file.replace('.png', '')
        path = os.path.join(folder, file)
        #this if statment was added to avoid the desktop.ini error
        if file.endswith('.png'):
            image = Image.open(path).convert('LA').resize((256, 256))
            #adding noise to images
            im2 = image.filter(ImageFilter.GaussianBlur(2))
            arr = np.array(im2)
            images.append((int(file_id), arr))
##    image.show()
    images.sort(key=lambda i: i[0])
    return np.array([v for _id, v in images])

start = time.perf_counter()

x_train = load_data('train')
y_train = pd.read_csv('y_train.csv')['infection']
#______________________________________________________________________
#This is a trial for future Assignments
####print(y_train)
### creating a noise with the same dimension as the dataset
##var = pd.DataFrame(y_train,dtype=float)['infection']
####var['infection'] = y_train.astype(float)
####print(var)
##mu, sigma = 0, 0.1 
##noise = np.random.normal(mu, sigma, [487,1]) 
##var2 = pd.DataFrame(noise)[0]
##var2 = var2.astype(float)
####print("noise signal:", var2)
##y_train= abs(var + var2)
##print(y_train)
##### normalize the data attributes
####norm = np.linalg.norm(y_train)
####y_train = y_train/norm
####print(y_train)

#Vanilla Example tempelate                                   
def build():
    img_in = Input(shape=(256, 256, 2))
    flattened = Flatten()(img_in)
    fc1 = Dense(128)(flattened)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(64)(fc1)
    fc2 = Dropout(0.3)(fc2)
    output = Dense(1, activation = 'sigmoid')(fc2)
    model = tf.keras.Model(inputs=img_in, outputs=output)
    return model

##My Model
##Modified AlexNet Architecture
model1 = tf.keras.models.Sequential([
    #these parameters where optimized after hundreds of test runs on-
    #-various combination of parameters either in the network architecture or training parameters
    tf.keras.layers.Conv2D(filters= 20,kernel_size=(16,16), strides=(5,1),padding='same', activation='relu', input_shape=(256,256,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
##just uncomment the following line to use vanilla tempelate and rename My model to any thing e.g. model1
##model = build()

#Optimizers to Use
##        1- RMSprop
##        2- Nadam
##        3- Adam
##        4- Adadelta

model = build()

#Model compile function to set up the optimzier's training, loss and accuracy metrics parameters.
model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.00001,beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',
        metrics=['BinaryAccuracy', 'AUC']
        )
callbacks = [
    ModelCheckpoint(
        filepath='model_{epoch}',
        save_best_only=False,
        verbose=1)
]
epochs = 10
batch_size = 64

#Training process
history = model.fit(x = x_train,
                    y = y_train,
                    batch_size = batch_size,
                    validation_split=0.3,
                    epochs=epochs,
                    verbose=2,
                    validation_freq=1,
                    )
model.save('models/MNIST_LeNet61.h5')
model.summary()

elapsed = time.perf_counter() - start

print('Elapsed %.3f mins.' % (elapsed/60))
t = time.localtime()
timestamp = time.strftime('%d-%b-%Y_%H-%M-%S', t)
tist =  timestamp

#Plotting the Network Architecture
tf.keras.utils.plot_model(
    model,
    to_file="models/Model_%d_epochs_%s.png"%(epochs,tist),
    show_shapes=True,
    show_layer_names=True,
    rankdir="LR",
    expand_nested=False,
    dpi=300,
)
###Testing the model on a new dataset
x_test = load_data('test')
y_test = model.predict(x_test)
print('test made')
#Shaping the submission file
y_test_df = pd.DataFrame()
y_test_df['id'] = np.arange(len(y_test))
y_test_df['infection'] = y_test.astype(float)
y_test_df.to_csv('submission.csv', index=False)

##Visualizing training and Validation

print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(20,15))    
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_auc'])
plt.title('model accuracy',fontsize=28)
plt.ylabel('accuracy',fontsize=28)
plt.xlabel('epoch',fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left',fontsize=18)
plt.grid(color='k', linewidth=0.1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.show()
plt.savefig('foo.png')
# summarize history for loss
plt.figure(figsize=(20,15))    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss',fontsize=28)
plt.ylabel('loss',fontsize=28)
plt.xlabel('epoch',fontsize=28)
plt.legend(['train', 'Validation'], loc='upper left',fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(color='k', linewidth=0.1)
#plt.show()
plt.savefig('foo2.png')
