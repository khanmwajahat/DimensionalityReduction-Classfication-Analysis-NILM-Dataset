#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
#import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler as scale
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


# In[4]:


ektest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\ES_TEST.txt', header=0 )
ebtest= pd.read_csv(r'C:\Users\kahnw\Desktop\Mcmaster\Intro to Machine Learning\project\EB_TEST.txt', header=0)


# In[7]:


data=pd.DataFrame(ebtest)
data


# In[51]:


x=ebtest.append(ektest)
df1 = pd.DataFrame(x)
data = df1.drop("Class",1)
y = df1["Class"] 


# In[52]:


#Backward Elimination
cols = list(data.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = data[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>1e-2000):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
df = pd.DataFrame(X_1)
X = df.drop("const",1)


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=4)
print(x_test)


# In[54]:


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


# In[55]:


print("Training time =",end1-start1)
print("Testing time =",end2-start2)


# In[ ]:




