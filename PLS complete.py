#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler as scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from timeit import default_timer as timer


# In[2]:


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[3]:


ektest= pd.read_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\EK_test_complete.xlsx', header=0, engine='openpyxl' )
estest= pd.read_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\ES_test_complete.xlsx', header=0, engine='openpyxl')


# In[4]:


x=estest.append(ektest)
df1 = pd.DataFrame(x)
data = df1.drop("Class",1)
y = df1["Class"]


# In[11]:


data_scaled=scale().fit_transform(data)
data_scaled= pd.DataFrame(data_scaled)
print(data_scaled)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(data_scaled, y, test_size=.3)


# In[9]:


classifier3=PLSRegression(n_components=2)
classifier3.fit(x_train, y_train)
y_pred = classifier3.predict(x_test)
#print(y_pred)

for i in range(len(y_pred)):
    if y_pred[i] < 3.5:
        y_pred[i]=3
    else:
        y_pred[i]=4
cm3=confusion_matrix(y_test, y_pred)
print(cm3)
print("Accuracy of PLSClassifier :" '', accuracy(cm3)*100)


# In[ ]:




