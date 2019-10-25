
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import seaborn as sns


# In[3]:


df = pd.read_csv('Iris.csv')


# In[4]:


df[0:10]


# In[5]:


#How many features are there
df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df['SepalLengthCm'].describe()

#Similarly for x2, x3, x4


# In[24]:


n = 150
sepal_list = df['SepalLengthCm']
sum_sepal = sum(sepal_list)
mean_sepal = sum_sepal / n
print(mean_sepal)


# In[36]:


total = 0
for row in df['SepalLengthCm']:
    total += float(row)
print(total/n)


# In[8]:


df['PetalWidthCm'].describe()


# In[10]:


df['Species'].describe()


# In[11]:


#histogram
plt.hist(df['SepalLengthCm'],bins=30)
plt.ylabel('No of times')
plt.show()

#Similarly for x2, x3, x4


# In[12]:


sns.boxplot(y=df['SepalLengthCm'])


# In[20]:


#one against all others

sns.boxplot(x=df['Species'],y=df['SepalWidthCm'])

