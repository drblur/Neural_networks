## importing necessary packages and modules
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.models import Sequential
import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from PIL import Image
np.random.seed(1212)

## importing the datasets, flattening them and adding the labels corresponding
train_path = os.getcwd()+'/mnist_train/Images/train/'
test_path = os.getcwd()+'/mnist_train/Images/test/'
files_train = os.listdir('mnist_train/Images/train')
files_test = os.listdir('mnist_train/Images/test')
train_labels = pd.read_csv('mnist_train/train.csv',sep=',')
test_labels = pd.read_csv('mnist_train/test.csv',sep=',')

train_X = []
test_X = []

for i in range(0,len(train_labels)):
    # o = []
    train_X.append(np.expand_dims(image.img_to_array(image.load_img(train_path+train_labels['filename'][i],
                    target_size=(28,28),grayscale=True)),axis=0).flatten())

for i in range(0,len(test_labels)):
    test_X.append(np.expand_dims(image.img_to_array(image.load_img(test_path+test_labels['filename'][i],
                target_size=(28,28),grayscale=True)),axis=0).flatten())
    # train_X.append(image.img_to_array(image.load_img(train_path+train_labels['filename'][i],target_size=(28,28))).flatten())
    # train_X.append(o)

# creating a data frame after flattening the images along side the labels
df = pd.DataFrame(train_X)
df.insert(0,"labels",train_labels['label'])

# splitting data for test and train
train_labels = df.loc[:,'labels']
train_features = df.iloc[:,1:]
from sklearn.model_selection import train_test_split
X_train, X_cv,y_train,y_cv = train_test_split(train_features, train_labels, test_size=0.1, random_state=1212)

X_train = X_train.as_matrix().reshape(len(X_train),784)
X_cv = X_cv.as_matrix().reshape(len(X_cv),784)
X_test = pd.DataFrame(test_X).as_matrix().reshape(len(test_X),784)

y_train_asis = y_train
y_cv_asis = y_cv


# data cleaning and normalising
X_train = X_train.astype('float32'); X_cv = X_cv.astype('float32')
X_train /= 255.0
X_cv /= 255.0
X_test /=255.0

# one hot encoding the labels
number_digits = 10
from keras.utils import to_categorical
y_train = to_categorical(y_train,number_digits)
y_cv = to_categorical(y_cv,number_digits)
y_cv[0]
y_train[2]

# now we fit the model to the preprocessed data
# parameters for the input
n_input = 784
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 100
n_hidden_5 = 200
number_digits = 10

inp = Input(shape=(784,))
 x = Dense(n_hidden_1, activation='relu', name='first_hidden_layer')(inp)
 x = Dense(n_hidden_2,activation='relu', name='second_hidden_layer')(x)
 x = Dense(n_hidden_3,activation='relu', name='third_hidden_layer')(x)
 x = Dense(n_hidden_4, activation='relu', name='fourth_hidden_layer')(x)
 x = Dense(n_hidden_5,activation='relu',name= 'fifth_hidden_layer')(x)
 output = Dense(number_digits,activation='softmax', name='output_layer')(x)

# building a model
model = Model(inp,output)
model.summary()

# hyper parameters
learning_rate = 0.1
train_epochs = 30
batch_size = 110
sgd = optimizers.SGD(lr=learning_rate)


#compiling the neural network
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

history1 = model.fit(X_train,y_train,batch_size=batch_size,epochs=train_epochs,verbose=2,validation_data=(X_cv,y_cv))

# lets use a different optimiser say adam and see if can increase the accuracy

model1 = Model(inp,output)
#compiling model 2
adam = keras.optimizers.Adam(lr=0.05)
model1.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
history2 = model1.fit(X_train,y_train,batch_size = 100,epochs=30,verbose=2,validation_data=(X_cv,y_cv))

# predicting the test set
test_pred = pd.DataFrame(model.predict(X_test,batch_size = 100))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_labels.insert(1,"label",test_pred)
test_labels.to_csv('prediction.csv')

test_pred1 = pd.DataFrame(model1.predict(X_test,batch_size = 100))
test_pred1 = pd.DataFrame(test_pred1.idxmax(axis=1))
test_labels.insert(1,"label",test_pred)
test_labels.to_csv('prediction3.csv')