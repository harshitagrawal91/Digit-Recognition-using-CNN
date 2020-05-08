import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from skimage.transform import resize

# keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adadelta, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import models


img_rows, img_cols = 50, 50
numClass=10
# Data loading
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_train = resize(X_train , (X_train.shape[0],50,50))
## splitting Data in train and test set##
X_train, X_test, Y_train, Y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
## Resize input images
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

##One hot encoding##
Y_train = to_categorical(Y_train, numClass)
Y_test_onehot = to_categorical(Y_test, numClass)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test_onehot.shape)

# Model training
model = Sequential()
model.add(Conv2D(activation='relu', padding='Same', filters=32, 
                 kernel_size=(5,5), input_shape=(50,50,1)))
model.add(Conv2D(activation='relu', padding='Same', filters=32, kernel_size=(5,5)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon = 1e-08, decay =0.0)
model.compile(optimizer= optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


epochs = 100 
batchSize = 32
datagen = ImageDataGenerator(rotation_range=10,zoom_range = 0.1,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=False, vertical_flip=False)

datagen.fit(X_train)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', verbose = 1, factor=0.5
                              ,patience=3, min_lr=0.00001)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batchSize),
                              epochs = epochs, validation_data = (X_test,Y_test_onehot),
                              verbose = 1, steps_per_epoch=X_train.shape[0]//batchSize
                              , callbacks=[reduce_lr])

score = model.evaluate(X_test, Y_test_onehot, verbose=1)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1]) 
##Saving Model In JSON Format
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

## Generating Confusion Matrix
y_pred=model.predict_classes(X_test)
print(confusion_matrix(Y_test, y_pred))