#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Install mlxtend library
get_ipython().system('pip install mlxtend')


# In[11]:


# Import necessary libraries

import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[12]:


# print the dataframe

titanic = pd.read_csv("Titanic.csv")
titanic


# In[13]:


titanic.info()


# In[14]:


titanic.isnull().sum()


# In[15]:


titanic.describe()


# ## Observation
# - No null values
# - all columns are object datatype
# - the table has 4 columns 2201x4

# In[21]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[22]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[23]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[24]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[28]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[29]:


df.info()


# # Apriori Algorithm

# In[30]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[31]:


frequent_itemsets.info()


# In[32]:


rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1.0)
rules


# In[ ]:




