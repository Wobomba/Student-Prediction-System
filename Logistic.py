#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries
from __future__ import division
import numpy as np
import pandas as pd
# additional imports
from time import time
from sklearn.metrics import f1_score


# In[3]:


student_data = pd.read_csv(r"D:\School\Year 3\Sem 2\Final Year Project\udacity-student-intervention-master\student-data.csv")
print("successfully read")


# In[5]:


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


# In[6]:


# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print ("\nFeature values:")
print (X_all.head())


# In[7]:


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# shuffle the data
X_all, y_all = shuffle(X_all, y_all, random_state=42)

# split the data into training and testing sets, 
# use `stratify` to maintain balance between classifications
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, 
                                                    test_size=0.24, random_state=42)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

# Check for imbalances
print ("Grad rate of the training set: {:.2f}%".format(100 * (y_train == 'yes').mean()))
print ("Grad rate of the testing set: {:.2f}%".format(100 * (y_test == 'yes').mean()))


# In[9]:


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))


# In[10]:


from sklearn.linear_model import LogisticRegression
peelt = LogisticRegression(solver='lbfgs',max_iter=500)
train_predict(peelt, X_train, y_train, X_test, y_test)


# In[11]:


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


# In[12]:


len(X_all.columns)


# In[14]:


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


# In[15]:


from sklearn.linear_model import LogisticRegression
predictor = LogisticRegression(solver='lbfgs',max_iter=500)
predictor.fit(X_train,y_train)


# In[16]:


predictor.predict(X_test)


# In[ ]:




