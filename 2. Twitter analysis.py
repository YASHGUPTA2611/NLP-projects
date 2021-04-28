#!/usr/bin/env python
# coding: utf-8

# # Making a Machine Learning model by using NLP to check that a Tweet is Racist or not.

# In[2]:


# Importing all important libraries which will be in use
import nltk
import numpy as np
import pandas as pd


# In[3]:


# Reading our dataset using pandas library
train = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\twitter analysis\train.csv')


# In[4]:


# Top 5 rows of our training data
train.head()


# In[10]:


# total rows and columns in training data
train.shape


# In[5]:


# Checking null values in training data
train.isnull().sum()


# In[7]:


# Reading test dataset
test = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\twitter analysis\test.csv')


# In[8]:


# Top 5 rows of our testing data
test.head()


# In[11]:


# total rows and columns in training data
test.shape


# In[9]:


# Checking null values in testing data
test.isnull().sum()


# In[7]:


import re


# In[8]:


# importing nltk objects which will be in further use.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmetizer=  WordNetLemmatizer()


# In[9]:


# storing values in col1 and col3 of tweet column for both training and testing data
col1 = train['tweet'].values
col3 = test['tweet'].values


# In[10]:


# Assining col2 and col4 to empty list.
col2 = []
col4 = []


# In[11]:


# using for loop, list compehension, nltk. We are removing unnecessary words and converting all texts into one paragraph for our training data.
for i in range(len(col1)):
    review = re.sub('[^a-zA-Z]', ' ', col1[i])
    review = review.lower()
    review = review.split()
    review = [lemmetizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    col2.append(review)


# In[12]:


col2


# In[13]:


# Using tf-idf to convert our text data into vectors 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(col2).toarray()


# In[14]:


X_train


# In[15]:


# storing output values in y variable
y = train['label'].values


# In[16]:


y


# ## Applying Naive Bayes Machine Learning Algorithm 

# In[17]:


from sklearn.naive_bayes import GaussianNB


# In[18]:


clf = GaussianNB()


# In[ ]:


tweet_analysis = clf.fit(X_train, y)


# In[ ]:


# using for loop, list compehension, nltk. We are removing unnecessary words and converting all texts into one paragraph for our testing data.
for i in range(len(col2)):
    review = re.sub('[^a-zA-Z]', ' ', col1[i])
    review = review.lower()
    review = review.split()
    review = [lemmetizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    col4.append(review)


# In[ ]:


col4


# In[ ]:


# converting text into vectors
X_test = tfidf.fit_transform(col4).toarray()


# In[ ]:


# storing predicted values in y_pred
y_pred = tweet_analysis.predict(X_test)


# In[ ]:


y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




