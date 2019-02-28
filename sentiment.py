#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from query import Database
from normalization import normalize


# In[2]:

db = Database()
data = []
data_labels = []

# with open("./data/pos_tweets.txt", encoding='utf-8') as f:
#     for i in f: 
#         data.append(i) 
#         data_labels.append('pos')

# with open("./data/neg_tweets.txt", encoding='utf-8') as f:
#     for i in f: 
#         data.append(i)
#         data_labels.append('neg')

for item in db.all_messages():
    data.append(item[0])
    data_labels.append(item[1])

data = normalize(data)
    
print(len(data))
# In[3]:

# In[4]:


vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase= False,
    # encoding='utf-8'
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_nd,
    data_labels,
    # train_size=0.80,
    random_state=1234
)


# In[14]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[15]:


log_model = log_model.fit(X=X_train, y=y_train)


# In[16]:


y_pred = log_model.predict(X_test)


# In[22]:


# import random
# j = random.randint(0,len(X_test)-7)
# for i in range(j,j+7):
#     print(y_pred[i])
#     ind = features_nd.tolist().index(X_test[i].tolist())
#     print(data[ind].strip())


# In[23]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[30]:

while True:
    test = input("type here: ")
    test = vectorizer.transform([test])
    result = log_model.predict(test)
    print(result)

# In[ ]:




