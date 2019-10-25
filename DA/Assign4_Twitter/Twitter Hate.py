
# coding: utf-8

# In[7]:


import pandas as pd
import re


# In[8]:


train=pd.read_csv("train.csv")


# In[9]:


train.head()


# In[10]:


train.drop("id",inplace=True,axis=1)


# In[5]:


import nltk
nltk.download()


# In[6]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def clean_sentences(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9^,!.\/']", " ", text)
    text = " ".join(text.split())
    text = " ".join(stemmer.stem(word) for word in text.split())
    return text


# In[11]:


x = train['tweet']
y = train['label']


# In[6]:


x = x.map(lambda a: clean_sentences(a))


# In[12]:


x.head()


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,random_state=42)


# In[14]:


x_train.head()


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


vectorizer = TfidfVectorizer(stop_words='english')


# In[17]:


x_train = vectorizer.fit_transform(x_train)


# In[18]:


x_test = vectorizer.transform(x_test)


# In[19]:


from sklearn.svm import LinearSVC


# In[20]:


model = LinearSVC(C=1.05, tol=0.5)


# In[21]:


model.fit(x_train,y_train)


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
confusion_matrix(y_test,model.predict(x_test))


# In[23]:


accuracy_score(y_test,model.predict(x_test))


# In[24]:


recall_score(y_test,model.predict(x_test))


# In[25]:


precision_score(y_test,model.predict(x_test))


# In[26]:


f1_score(y_test,model.predict(x_test))

