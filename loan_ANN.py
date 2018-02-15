import numpy as np
import pandas as pd
import keras
from fancyimpute import KNN, NuclearNormMinimization
import time
import random

# set seed
random.seed(2)

## importing the dataset for the problem
dataset = pd.read_csv("train.csv")

## dropping features which has more than 5% of missing values
dataset = dataset.drop(dataset.columns[[0,4,6,10,13]],axis=1)
dataset['DOB'] = pd.to_datetime(dataset['DOB'],format="%d/%m/%y")
dataset['Lead_Creation_Date'] = pd.to_datetime(dataset['Lead_Creation_Date'], format="%d/%m/%y")
dataset.isnull().sum()
y = dataset.iloc[:,16].values
dataset = dataset.drop(dataset.columns[[16]],axis=1)
##converting DOB to years
age=[]
lead_creation=[]
for i in range(0,len(dataset)):
    if time.localtime().tm_year - dataset['DOB'][i].year >0:
        age.append((time.localtime().tm_year - dataset['DOB'][i].year))
        lead_creation.append((dataset['Lead_Creation_Date'][i].dayofyear))
    else:
        age.append(np.mean(age))
        lead_creation.append((dataset['Lead_Creation_Date'][i].dayofyear))



dataset = dataset.drop(dataset.columns[[1,2]],axis=1)
dataset.insert(1,"DOB",age)
dataset.insert(2,"Lead_Creation_Date",lead_creation)
dataset.nunique()


## imputing the features
dataset['City_Category'].fillna(value="mode",inplace= True)
dataset['Employer_Category1'].fillna(value="mode",inplace= True)
dataset['Employer_Category2'].fillna(value="mode",inplace= True)
dataset['Existing_EMI'].fillna(value=0,inplace = True)
dataset['Loan_Amount'].fillna(value=np.min(dataset['Loan_Amount']),inplace = True)
dataset['Loan_Period'].fillna(value=np.min(dataset['Loan_Period']),inplace = True)
dataset['Interest_Rate'].fillna(value=np.min(dataset['Interest_Rate']),inplace = True)
dataset['EMI'].fillna(value=np.min(dataset['EMI']),inplace = True)
dataset['Primary_Bank_Type'].fillna(value="mode",inplace = True)


#Encoding the categorical data for country and gender
gender = pd.get_dummies(dataset['Gender'],prefix='Gender').iloc[:,1:]
# lead_day = pd.get_dummies(dataset['Lead_Creation_Date'],prefix="Lead_day").iloc[:,1:]
city_category = pd.get_dummies(dataset.City_Category, prefix="City_Cat").iloc[:,1:]
emp_category1 = pd.get_dummies(dataset.Employer_Category1, prefix="emp_Cat1").iloc[:,1:]
emp_category2 = pd.get_dummies(dataset.Employer_Category2, prefix="emp_Cat2").iloc[:,1:]
primary_bank_type = pd.get_dummies(dataset.Primary_Bank_Type,prefix="Pri_bank_code").iloc[:,1:]
contacted = pd.get_dummies(dataset['Contacted'],prefix="Contacted").iloc[:,1:]
source_Cat = pd.get_dummies(dataset.Source_Category,prefix="Src_Cat").iloc[:,1:]
variable_1 = pd.get_dummies(dataset['Var1'],prefix="Var").iloc[:,1:]

##removing the features from which dummy variables were extracted
dataset = dataset.drop(dataset.columns[[0,3,4,5,7,8,9,15]],axis=1)
dataset = pd.concat([dataset,city_category],axis=1)
dataset = pd.concat([dataset,emp_category1],axis=1)
dataset = pd.concat([dataset,emp_category2],axis=1)
dataset = pd.concat([dataset,contacted],axis=1)
dataset = pd.concat([dataset,gender],axis=1)
# dataset = pd.concat([dataset,lead_day],axis=1)
dataset = pd.concat([dataset,variable_1],axis=1)
dataset = pd.concat([dataset,primary_bank_type],axis=1)
dataset = pd.concat([dataset,source_Cat],axis=1)

##separating X from y
X = dataset.iloc[:,].values

##Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Scaling the training and testing data so that Gradient descent can be faster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
classifier.add(Dense(14,activation='relu',kernel_initializer='uniform',input_shape=(32,)))


 ## adding another hidden layer
classifier.add(Dense(14,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(14,activation='relu',kernel_initializer='uniform'))

 ## adding the output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid')) # if we are dealing with more than 2 categories
                                                                           # in this case the output changes and activation
                                                                           # function changes to softmax
 ## Compiling our ANN
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) # adam refers to stochastic gradient descent
                                                                                    # in case there are more than 2 categories
                                                                                    #we use category_crossentropy
 ## training the ANN
classifier.fit(X_train,y_train,batch_size=15,nb_epoch=10)


 ## predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) ## converting the probabilities to boolean with a threshold of 0.5
##Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# importing the test set and pre_process the same

dataset = pd.read_csv("test.csv")

## dropping features which has more than 5% of missing values
dataset = dataset.drop(dataset.columns[[0,4,6,10,13]],axis=1)
dataset['DOB'] = pd.to_datetime(dataset['DOB'],format="%d/%m/%y")
dataset['Lead_Creation_Date'] = pd.to_datetime(dataset['Lead_Creation_Date'], format="%d/%m/%y")
dataset.isnull().sum()
# y = dataset.iloc[:,16].values
# dataset = dataset.drop(dataset.columns[[16]],axis=1)
##converting DOB to years
age=[]
lead_creation=[]
for i in range(0,len(dataset)):
    if time.localtime().tm_year - dataset['DOB'][i].year >0:
        age.append((time.localtime().tm_year - dataset['DOB'][i].year))
        lead_creation.append((dataset['Lead_Creation_Date'][i].dayofyear))
    else:
        age.append(np.mean(age))
        lead_creation.append((dataset['Lead_Creation_Date'][i].dayofyear))



dataset = dataset.drop(dataset.columns[[1,2]],axis=1)
dataset.insert(1,"DOB",age)
dataset.insert(2,"Lead_Creation_Date",lead_creation)
dataset.nunique()


## imputing the features
dataset['City_Category'].fillna(value="mode",inplace= True)
dataset['Employer_Category1'].fillna(value="mode",inplace= True)
dataset['Employer_Category2'].fillna(value="mode",inplace= True)
dataset['Existing_EMI'].fillna(value=0,inplace = True)
dataset['Loan_Amount'].fillna(value=np.min(dataset['Loan_Amount']),inplace = True)
dataset['Loan_Period'].fillna(value=np.min(dataset['Loan_Period']),inplace = True)
dataset['Interest_Rate'].fillna(value=np.min(dataset['Interest_Rate']),inplace = True)
dataset['EMI'].fillna(value=np.min(dataset['EMI']),inplace = True)
dataset['Primary_Bank_Type'].fillna(value="mode",inplace = True)


#Encoding the categorical data for country and gender
gender = pd.get_dummies(dataset['Gender'],prefix='Gender').iloc[:,1:]
# lead_day = pd.get_dummies(dataset['Lead_Creation_Date'],prefix="Lead_day").iloc[:,1:]
city_category = pd.get_dummies(dataset.City_Category, prefix="City_Cat").iloc[:,1:]
emp_category1 = pd.get_dummies(dataset.Employer_Category1, prefix="emp_Cat1").iloc[:,1:]
emp_category2 = pd.get_dummies(dataset.Employer_Category2, prefix="emp_Cat2").iloc[:,1:]
primary_bank_type = pd.get_dummies(dataset.Primary_Bank_Type,prefix="Pri_bank_code").iloc[:,1:]
contacted = pd.get_dummies(dataset['Contacted'],prefix="Contacted").iloc[:,1:]
source_Cat = pd.get_dummies(dataset.Source_Category,prefix="Src_Cat").iloc[:,1:]
variable_1 = pd.get_dummies(dataset['Var1'],prefix="Var").iloc[:,1:]

##removing the features from which dummy variables were extracted
dataset = dataset.drop(dataset.columns[[0,3,4,5,7,8,9,15]],axis=1)
dataset = pd.concat([dataset,city_category],axis=1)
dataset = pd.concat([dataset,emp_category1],axis=1)
dataset = pd.concat([dataset,emp_category2],axis=1)
dataset = pd.concat([dataset,contacted],axis=1)
dataset = pd.concat([dataset,gender],axis=1)
# dataset = pd.concat([dataset,lead_day],axis=1)
dataset = pd.concat([dataset,variable_1],axis=1)
dataset = pd.concat([dataset,primary_bank_type],axis=1)
dataset = pd.concat([dataset,source_Cat],axis=1)

##separating X from y
X = dataset.iloc[:,].values


## Scaling the training and testing data so that Gradient descent can be faster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
X_test = sc.fit_transform(X)

## predict

y_pred = classifier.predict(X_test)
submission = pd.read_csv("test.csv")
submission_final = pd.DataFrame(submission['ID'])
submission_final.insert(1,'Approved',y_pred)
submission_final.to_csv("result4.csv")
##Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) # the accuracy on the test is close to 86%

 ## checking with a new observation

from keras.models import Sequential #Implying a sequential ANN
from keras.layers import Dense  #used to creaate layers in our ANNs
from keras.wrappers.scikit_learn import KerasClassifier # It is a keras wrapper for scikit learn that will enable us
                                                        #  to use cross_validation function of
                                                        #scikit learn
from sklearn.model_selection import cross_val_score # importing the cross validation fuction required for K fold cv
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(15, activation='relu', kernel_initializer='uniform', input_shape=(32,)))
    classifier.add(Dense(15, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(15, activation='relu', kernel_initializer='uniform'))
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


