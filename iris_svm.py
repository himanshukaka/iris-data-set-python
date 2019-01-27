# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 07:57:56 2018

@author: HIMANSHU
"""


import numpy as np
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('Iris.csv')

corr = dataset.iloc[:,1:5].corr()
sns.heatmap(corr, annot = True)

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, [5]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder() 
y= labelencoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate Test Prediction
print(classifier.score(X_test,y_test.ravel()))


#plotting
df_cm = pd.DataFrame(cm, index = [i for i in np.unique(y)],
                  columns = [i for i in np.unique(y)])
plt.figure(figsize = (5,5))
sn.heatmap(df_cm, annot=True)



