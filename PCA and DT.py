#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler as scale
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from timeit import default_timer as timer


# In[73]:


ektest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\ES_TEST.txt', header=0 )
ebtest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\EB_TEST.txt', header=0)
print (ektest)


# In[74]:


x=ebtest.append(ektest)
df1 = pd.DataFrame(x)
data = df1.drop("Class",1)
y = df1["Class"]
print (y)


# In[75]:


data_scaled=scale().fit_transform(x)
from sklearn import decomposition
pca = decomposition.PCA(n_components=5)
pca_x = pca.fit_transform(data_scaled)
pd.DataFrame(pca_x)


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=.25)
pd.DataFrame(x_test)


# In[77]:


# training set result without feature selection
classifier = DecisionTreeClassifier()
#training
start1 = timer()
classifier.fit(x_train, y_train)
end1 = timer()
#testing
start2 = timer()
y_pred = classifier.predict(x_test)
end2 = timer()
confusion_matrix(y_test, y_pred)


# In[78]:


print("Training time =",end1-start1)
print("Testing time =",end2-start2)


# In[ ]:





# In[ ]:




