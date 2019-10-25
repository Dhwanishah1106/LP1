
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('diabetes.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data['Pregnancies'].describe()


# In[7]:


data.dtypes


# In[8]:


#train = np.array(data.iloc[0:600])
#test = np.array(data.iloc[600:768])
#train.shape


# In[11]:


#test.shape
y = data["Outcome"]
X = data.drop(["Outcome"], axis=1)


# In[12]:


from sklearn.model_selection import train_test_split


# In[57]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=5)


# In[58]:


#train_x = train[train.columns[:8]]
#test_x = test[test.columns[:8]]


# In[59]:


#train_y = train['Outcome']
#test_y = test['Outcome']


# In[60]:


from sklearn.naive_bayes import GaussianNB


# In[61]:


model = GaussianNB()


# In[62]:


model.fit(train_x, train_y)


# In[63]:


predict_y = model.predict(test_x)
#print(predict_y)


# In[64]:


from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[65]:


accuracy = accuracy_score(predict_y, test_y) * 100


# In[66]:


print("Accuracy : ", accuracy)


# In[67]:


print("Classification report : ")
print(metrics.classification_report(test_y, predict_y))


# In[68]:


print("Confusion matrix : ")
print(metrics.confusion_matrix(test_y, predict_y))

