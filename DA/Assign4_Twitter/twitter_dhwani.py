
# coding: utf-8

# In[24]:


import pandas as pd


# In[25]:


train = pd.read_csv('train.csv')


# In[26]:


train.drop("id", inplace=True, axis=1)


# In[27]:


x = train['tweet']
y = train['label']


# In[28]:


x.head()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y, random_state=5)


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


vectorizer = TfidfVectorizer(stop_words='english')


# In[33]:


train_x = vectorizer.fit_transform(train_x)


# In[34]:


test_x = vectorizer.transform(test_x)


# In[35]:


from sklearn.svm import LinearSVC


# In[36]:


model = LinearSVC(C=2,tol=0.5)


# In[37]:


model.fit(train_x, train_y)


# In[38]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[39]:


confusion_matrix(test_y, model.predict(test_x))


# In[40]:


accuracy_score(test_y, model.predict(test_x))


# In[41]:


precision_score(test_y, model.predict(test_x))


# In[42]:


f1_score(test_y, model.predict(test_x))

