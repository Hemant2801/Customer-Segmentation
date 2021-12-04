#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# # Data collection and analysis

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Customer segmentation/Mall_Customers.csv')


# In[3]:


# print the first 5 rows of the dataset
df.head()


# In[4]:


# print the last 5 rows of the dataset
df.tail()


# In[5]:


# shape of the dataset
df.shape


# In[6]:


# getting some information of the dataset
df.info()


# In[7]:


# checking for missing values
df.isnull().sum()


# Choosing the annual income and spending score column

# In[8]:


X = df.iloc[:,[3,4]].values


# In[9]:


print(X)


# Choosing the number of clusters
# 
# WCSS ---> Within Clusters Sum of Squares
# 
# WCSS is the sum of squared distance between each point and the centroid in a cluster.

# In[10]:


# finding the WCSS value for different numbers of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)


# In[11]:


# plot an elbow graph
sns.set_style(style = 'darkgrid')
plt.figure(figsize = (5,5))
plt.plot(range(1,11), wcss)
plt.title('ELBOW POINT GRAPH')
plt.xlabel('NO. OF CLUSTERS')
plt.ylabel('WCSS')
plt.show()


# The optimum no. of clusters are 5

# # Training the K-Means clustering model

# In[12]:


kmeans = KMeans(n_clusters = 5, init='k-means++', random_state = 0)


# In[13]:


# Retuen a label for each data point based on their cluster
Y = kmeans.fit_predict(X)
print(Y)


# # Visualising all the clusters

# In[14]:


# plotting all the clusters and their centroids
plt.figure(figsize = (8,8))
plt.scatter(X[Y == 0,0], X[Y == 0,1], s = 50, c = 'green', label = 'Cluster_1') # s = size of the dots
plt.scatter(X[Y == 1,0], X[Y == 1,1], s = 50, c = 'red', label = 'Cluster_2') # c = color of the dots
plt.scatter(X[Y == 2,0], X[Y == 2,1], s = 50, c = 'blue', label = 'Cluster_3')
plt.scatter(X[Y == 3,0], X[Y == 3,1], s = 50, c = 'yellow', label = 'Cluster_4')
plt.scatter(X[Y == 4,0], X[Y == 4,1], s = 50, c = 'violet', label = 'Cluster_5')

#plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroid')

plt.title('CUSTOMER GROUPS')
plt.xlabel('ANNUAL INCOME')
plt.ylabel('SPENDING SCORE')
plt.show()


# In[ ]:




