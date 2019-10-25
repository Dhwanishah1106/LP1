
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


df.hist()
plt.show()


# In[6]:


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[8]:


train_mean_pos = X_train[y_train == 1].mean()
train_std_pos = X_train[y_train == 1].std()
train_mean_neg = X_train[y_train == 0].mean()
train_std_neg = X_train[y_train == 0].std()


# In[9]:


summary = {"train_mean_pos":train_mean_pos.tolist(),
            "train_mean_neg":train_mean_neg.tolist(),
             "train_std_pos":train_std_pos.tolist(),
              "train_std_neg":train_std_neg.tolist()
          }


# In[10]:


summary


# In[11]:


def cond_prob(x, mean, std):
    variance = std*std
    p = 1/(np.sqrt(2*np.pi*variance)) * np.exp((-(x-mean)**2) / (2*variance))
    return p


# In[12]:


def posterior_prob(row, summary):
    pos_prob = len(X_train[y_train == 1]) / (len(X_train))
    neg_prob = len(X_train[y_train == 0]) / (len(X_train))
    
    for i in range(0, len(row)):
        pos_prob = pos_prob * cond_prob(row[i], train_mean_pos[i], train_std_pos[i])
    for i in range(0, len(row)):
        neg_prob = neg_prob * cond_prob(row[i], train_mean_neg[i], train_std_neg[i])
    return [pos_prob, neg_prob]


# In[13]:


predicted_raw = []
for row in X_test.values.tolist():
    predicted_raw.append(posterior_prob(row, summary))


# In[14]:


predicted_raw


# In[15]:


predictions = []
for row in predicted_raw:
    if(row[0] > row[1]):
        predictions.append(1)
    else:
        predictions.append(0)


# In[18]:


confusion_matrix(predictions, y_test)


# In[20]:


accuracy_score(predictions, y_test)

