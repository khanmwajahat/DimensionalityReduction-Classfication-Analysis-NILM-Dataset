#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from scipy import linalg as LA
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler as scale
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


# In[24]:


ektest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\ES_TEST.txt', header=0 )
ebtest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\EB_TEST.txt', header=0)
print (ektest)


# In[25]:


x=ebtest.append(ektest)
df1 = pd.DataFrame(x)
data = df1.drop("Class",1)
y = df1["Class"]
print (y)


# In[26]:


data_scaled=scale().fit_transform(x)
from sklearn import decomposition
pca = decomposition.PCA(n_components=5)
pca_x = pca.fit_transform(data_scaled)
pd.DataFrame(pca_x)


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=.25)
pd.DataFrame(x_test)


# In[28]:



# training set result without feature selection
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
#training
start1 = timer()
lda.fit(x_train, y_train)
end1 = timer()
#testing
start2 = timer()
y_pred = lda.predict(x_test)
end2 = timer()
confusion_matrix(y_test, y_pred)


# In[29]:


print("Training time =",end1-start1)
print("Testing time =",end2-start2)


# In[ ]:




