#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn


# In[4]:


df = pd.read_csv('Mobile Price Factors.csv')


# In[5]:


df.head()


# In[7]:


df.shape


# In[9]:


df.columns


# In[11]:


df.dtypes


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.corr()


# In[16]:


df.nunique()


# In[17]:


df.isna().sum()


# In[18]:


df.duplicated()


# In[22]:


df1=df.drop_duplicates()


# In[24]:


df1.reset_index(inplace=True,drop=True)


# In[25]:


df1


# In[26]:


df1['Front camera count'].value_counts()


# In[27]:


df1['Front camera count'].nunique()


# # Visualization

# In[55]:


sns.set_color_codes('dark')


# In[63]:


sns.scatterplot(x='Battery Power',y='Mobile Price',hue='Front camera count',data=df1)


# In[64]:


sns.scatterplot(x='Processor Speed',y='Mobile Price',hue='Front camera count',data=df1)


# In[65]:


sns.scatterplot(x='Mobile Brand',y='Mobile Price',hue='Front camera count',data=df1)


# In[66]:


sns.boxplot(df1['Mobile Brand'])


# In[67]:


sns.boxplot(df1['Mobile Price'])


# In[61]:


df1=df1[df1["Mobile Price"]<45000]


# # data modelling

# In[68]:


df1.columns


# In[72]:


X=df1[['Mobile Brand']]


# In[73]:


y=df1[['Mobile Price']]


# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[76]:


X_train.shape


# In[77]:


y_train.shape


# In[79]:


y_test.shape


# In[81]:


y_test.shape


# In[87]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[88]:


lr.fit(X_train,y_train)


# In[89]:


y_pred=lr.predict(X_test)


# In[90]:


y_pred


# In[91]:


y_train


# In[99]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# # model 2

# In[92]:


df1.columns


# In[93]:


X=df1[['Battery Power','Mobile Brand','Internal Memory',]]


# In[94]:


y=df1[['Mobile Price']]


# In[95]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[96]:


lr2=LinearRegression()


# In[97]:


lr2.fit(X_train,y_train)


# In[98]:


y_pred=lr2.predict(X_test)


# In[101]:


mean_squared_error(y_test,y_pred)


# In[ ]:


# hence from both the models we predict the same value.

