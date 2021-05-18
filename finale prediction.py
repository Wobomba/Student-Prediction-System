#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import libraries
from __future__ import division
import numpy as np
import pandas as pd
# additional imports
from time import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[10]:


student_data = pd.read_csv(r"D:\School\Year 3\Sem 2\Final Year Project\udacity-student-intervention-master\student-data.csv")
print("successfully read")


# In[11]:


#exploring the data from the data set
#accessing the features to get the number of students
n_students = student_data.shape[0] 
#obtaining the number of students
assert n_students == student_data.passed.count()
#accessing the number of features in the dataset minus the target feature, passed
n_features = student_data.shape[1] - 1
#obtaining the total number of features in the dataset
assert n_features == len(student_data.columns[student_data.columns != 'passed'])
#accessing the students that passed in the previous examination
n_passed = sum(student_data.passed.map({'no': 0, 'yes': 1}))
#obtaining the number of students who passed the examination
assert n_passed == len(student_data[student_data.passed == 'yes'].passed)
#computing the number of students who failed the previous examination
n_failed = n_students - n_passed
#computing the rate of those students who will graduate to the next level
grad_rate = n_passed/float(n_students)
print ("Total number of students: {}".format(n_students))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))
print ("Number of features: {}".format(n_features))
print ("Promotion rate of the class: {:.2f}%".format(100 * grad_rate))


# In[12]:


#preparation of the data for modeling, training, and testing
#extract feature (X) and target (y) columns to correct features that have non-numeric data.
feature_cols = list(student_data.columns[:-1])  
# all columns but last are features
#print(feature_cols)
target_col = student_data.columns[-1]  # last column is the target/label
print ("Feature column(s):-\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print ("\nFeature values:-")
print (X_all.head())  # print the first 5 rows


# In[13]:


# Preprocess feature columns
#conversion of the non-numeric features having categorical data into binary values
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    print('\n', outX, '\n')

    # check each column
    for col, col_data in X.iteritems():
        # if data type is non-numeric, try to replace all yes or no values with 1 or 0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # this should change the data type for yes/no columns to int

        # if they are still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col) 

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
y_all = y_all.replace(['yes', 'no'], [1, 0])
print ("Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[15]:


#splitting and training the dataset
# computing how many training vs test samples you want
num_all = student_data.shape[0]  
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# select features (X) and corresponding labels (y) for the training and test sets
# Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset(sampling the data)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size=num_test,
                                                    train_size=num_train,
                                                    random_state=100)
assert len(y_train) == 300
assert len(y_test) == 95
print ("Training set: {} samples".format(X_train.shape[0]))
print ("Test set: {} samples".format(X_test.shape[0]))


# In[16]:


from sklearn.linear_model import LogisticRegression
#instance of the model
predictor = LogisticRegression(solver='lbfgs',max_iter=500)
predictor.fit(X_train,y_train)


# In[17]:


predictor.predict(X_test)


# In[18]:


#probability of the student failing to that of passing
predictor.predict_proba(X_test)


# In[19]:


import pickle
import joblib
filename = 'newton_model.pkl'
joblib.dump(predictor, filename)


# In[ ]:




