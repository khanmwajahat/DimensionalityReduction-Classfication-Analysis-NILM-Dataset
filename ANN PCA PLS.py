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


ektest= pd.read_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\EK_TEST.xlsx', header=0, engine='openpyxl' )
estest= pd.read_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\ES_TEST.xlsx', header=0, engine='openpyxl')


# In[4]:


x=estest.append(ektest)
df1 = pd.DataFrame(x)
data = df1.drop("Class",1)
y = df1["Class"]
print (data)


# In[5]:


data= pd.DataFrame(data)
data.to_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\data_unscaled.xlsx', header=0)


# In[6]:


data_scaled=scale().fit_transform(data)
data_scaled= pd.DataFrame(data_scaled)
data_scaled.to_excel(r'C:\Users\kahnw\Desktop\Mcmaster\multivariate stats\data_scaled.xlsx', header=0)


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(data_scaled, y, test_size=.3)
pd.DataFrame(x_test)


# In[8]:


classifier1 = MLPClassifier(hidden_layer_sizes=(3), max_iter=10,activation = 'relu',solver='adam',random_state=1)


# In[9]:


classifier1.fit(x_train, y_train)
y_pred = classifier1.predict(x_test)
cm1=confusion_matrix(y_test, y_pred)


# In[10]:


print("Accuracy of MLPClassifier :" '', accuracy(cm1)*100)


# In[11]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=10)
pca_x = pca.fit_transform(data_scaled)


# In[12]:


x_train_pca, x_test_pca, y_train, y_test = train_test_split(pca_x, y, test_size=.3)


# In[13]:


classifier2 = MLPClassifier(hidden_layer_sizes=(3), max_iter=10,activation = 'relu',solver='adam',random_state=1)
classifier2.fit(x_train_pca, y_train)
y_pred = classifier2.predict(x_test_pca)
cm2=confusion_matrix(y_test, y_pred)
print(cm2)
print("Accuracy of MLPClassifier with PCA :" '', accuracy(cm2)*100)


# In[14]:


x_train_PLS, x_test_PLS, y_train, y_test = train_test_split(data_scaled, y, test_size=.25)
classifier3=PLSRegression(n_components=2)
classifier3.fit(x_train_PLS, y_train)
y_pred = classifier3.predict(x_test_PLS)
#print(y_pred)

for i in range(500):
    if y_pred[i] < 3.5:
        y_pred[i]=3
    else:
        y_pred[i]=4
cm3=confusion_matrix(y_test, y_pred)
print(cm3)
print("Accuracy of PLSClassifier :" '', accuracy(cm3)*100)


# In[ ]:




