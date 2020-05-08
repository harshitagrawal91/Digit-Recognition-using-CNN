import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import Adam,SGD,Adadelta, RMSprop
from skimage.transform import resize
from keras.utils.np_utils import to_categorical
def predictDigits(X_test,Y_test):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon = 1e-08, decay =0.0)
    loaded_model.compile(optimizer= optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    score = loaded_model.evaluate(X_test, Y_test_one, verbose=1)
    # prob=loaded_model.predict(X_test,batch_size=None, verbose=1)
    Y_Predict=loaded_model.predict_classes(X_test)
    for predict in Y_Predict:
        print(predict, end=" ")
    print('\nTest loss:', score[0]) 
    print('Test accuracy:', score[1]) 
    return Y_Predict

##Main Program##    
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
img_rows, img_cols = 50, 50
numClass=10
X_test = resize(X_test, (X_test.shape[0],50,50))
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
##One hot Encoding##
Y_test_one = to_categorical(Y_test, numClass)

## Generating Confusion Matrix
y_pred=predictDigits(X_test,Y_test_one)
print(y_pred)
print(confusion_matrix(Y_test, y_pred))


