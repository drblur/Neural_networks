import numpy as np
import pandas as pd
import keras
## importing the dataset for the problem
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encoding the categorical data for country and gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enccoder_X_1 = LabelEncoder()
X[:,1] = label_enccoder_X_1.fit_transform(X[:,1])# index is 1 because we are encoding Country
label_enccoder_X_2 = LabelEncoder()
X[:,2] = label_enccoder_X_2.fit_transform(X[:,2]) # index is 2 as we are encoding gender
onehotencoder =OneHotEncoder(categorical_features=[1]) #it is the index of the feature for which dummy needs to be
                                                       # created
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #taking all the columns excluding one of the dummy variable to avoid dummy variable trap

##Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Scaling the training and testing data so that Gradient descent can be faster
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

## building the ANN (commenting the lines of building ANN and testing agains one observation)
import keras
from keras.models import Sequential #Implying a sequential ANN
from keras.layers import Dense  #used to creaate layers in our ANNs

## initializing the ANN
classifier = Sequential()

## adding the input layer and the first hidden layer
classifier.add(Dense(6,activation='relu',kernel_initializer='uniform',input_dim=11))

## adding another hidden layer
classifier.add(Dense(6,activation='relu',kernel_initializer='uniform'))

## adding the output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid')) # if we are dealing with more than 2 categories
                                                                          # in this case the output changes and activation
                                                                          # function changes to softmax
## Compiling our ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # adam refers to stochastic gradient descent
                                                                                   # in case there are more than 2 categories
                                                                                   #we use category_crossentropy
## training the ANN
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=5) # accuracy of about 86.50%

## predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) ## converting the probabilities to boolean with a threshold of 0.5

##Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) # the accuracy on the test is close to 86%

## checking with a new observation

new_observation = classifier.predict(sc.fit_transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
## using [[]] will take the elements in horizontal vectors

## as the training set was in the same 2 dimensional format
new_observation = (new_prediction > 0.5)

print new_observation # it returns false that is the he is not moving out

########################################################K-fold cross-validation########################################

##we now evaluate our ANN with K fold cross validation to see if there is any variance in the model and if the model
# really has the accuracy of over 85%
## we also tune the parameters for the Cross-validation to see what are the parameters that give us the best accuracy.

from keras.models import Sequential #Implying a sequential ANN
from keras.layers import Dense  #used to creaate layers in our ANNs
# from keras.wrappers.scikit_learn import KerasClassifier # It is a keras wrapper for scikit learn that will enable us
#                                                         #  to use cross_validation function of
#                                                         #scikit learn
# from sklearn.model_selection import cross_val_score # importing the cross validation fuction required for K fold cv
#
#
# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_dim=11))
#     classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
#     classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10,epochs = 100)
# accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
# model_meam_accuracy = accuracies.mean()
# model_variiance = accuracies.std()

##################################################parameter tuning using Grid search####################################

## tuning the parameters for our ANN using GridSearch

# from keras.models import Sequential #Implying a sequential ANN
# from keras.layers import Dense  #used to creaate layers in our ANNs
# from keras.wrappers.scikit_learn import KerasClassifier # It is a keras wrapper for scikit learn that will enable us
#                                                         #  to use cross_validation function of
#                                                         #scikit learn
# from sklearn.model_selection import cross_val_score # importing the cross validation fuction required for K fold cv
# from sklearn.model_selection import GridSearchCV
#
#
# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_dim=11))
#     classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
#     classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn=build_classifier)
# parameters = {'batch_size':[25,34],'epochs':[100,500],'optimizer':['adam','rmsprop']}
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy',cv=10)
# grid_search = grid_search.fit(X_train,y_train)
#
# best_params = grid_search.best_params_
# best_accuracy = grid_search.best_score_
#
# print best_params # we got 100 epochs, 'adam' and batch size of 34 as our optimal parameters
# print best_accuracy # accuracy of 85.05%

## tuning the model further with different parameters

from keras.models import Sequential #Implying a sequential ANN
from keras.layers import Dense  #used to creaate layers in our ANNs
from keras.wrappers.scikit_learn import KerasClassifier # It is a keras wrapper for scikit learn that will enable us
                                                        #  to use cross_validation function of
                                                        #scikit learn
from sklearn.model_selection import cross_val_score # importing the cross validation fuction required for K fold cv
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(8, activation='relu', kernel_initializer='uniform', input_dim=11))
    classifier.add(Dense(8, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[34,39,42],'epochs':[100,125,155,200],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print best_params # we got 155 epochs, 'rmsprop' and batch size of 39 as our optimal parameters
print best_accuracy # accuracy of 85.25%


