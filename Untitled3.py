#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from auth import db


# In[2]:


def connect():
        connection = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' password='%s'" \
            %(db['dbname'], db['user'], db['host'], db['password'])
        )
        connection.autocommit = True
        cursor = connection.cursor()
        return cursor


# In[3]:


def get_data(sql):
    cursor = connect()
    cursor.execute(sql)
    reviews = DataFrame(cursor.fetchall())
    reviews.rename(columns={0:'text', 1:'label'}, inplace=True)
    cursor.close()
    return reviews


# In[4]:


sql = "SELECT stc.sentence, snt.sentiment FROM sentences stc                JOIN sentiment snt ON stc.id = snt.sentenceid                WHERE stc.id IN (SELECT sentenceid FROM sentiment)                AND snt.sentiment != 0"
data = get_data(sql)


# In[5]:


from unicodedata import normalize as nm
def remover_acentos(txt):
    return nm('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')


# In[52]:


def prepare_data(data):
    sentences = []
    for index, r in data.iterrows():
        text = r.text

        if r['label'] == 1:
            sentences.append([text,'positivo'])
        else:
            sentences.append([text,'negativo'])
    
    df = DataFrame(sentences, columns=['text', 'label'])
    return df


# In[53]:


processed_data = prepare_data(data)


# In[54]:


x = processed_data.text
y = processed_data.label


# In[97]:


x_train, x_validation_and_test, y_train, y_validation_and_test =     train_test_split(x, y, test_size=.02, random_state=2000)
x_validation, x_test, y_validation, y_test =     train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=2000)


# In[98]:


from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[99]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BaseDiscreteNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# In[100]:


vectorizer = TfidfVectorizer()
vectorizer.set_params(ngram_range=(1, 1), max_features=1500)

lr = LogisticRegression()
checker_pipeline = Pipeline([('vectorizer', vectorizer),('classifier', lr)])


# In[101]:


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("score de precisão: %f"%(accuracy*100))
    return sentiment_fit


# In[102]:


pred = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)


# In[78]:

while True:
    scan = input('type: ')
    if scan == 'x':
        break
    print(pred.predict([scan]))


# In[ ]:




