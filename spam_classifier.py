#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[3]:


# Load the dataset
data = pd.read_csv('emails.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.head()


# In[7]:


#Printing all the columns present in data
data.columns


# In[8]:


# A closer look at the data types present in the data
data.dtypes


# In[9]:


data.isnull().sum()


# In[10]:


# Apply label encoding to convert categorical variables
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])


# In[11]:


# Split features and labels
X = data.drop('Prediction', axis=1)
y = data['Prediction']


# In[12]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()


# In[14]:


# Train the classifier
classifier.fit(X_train, y_train)


# In[15]:


# Make predictions on the test set
y_pred = classifier.predict(X_test)


# In[16]:


# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)


# In[17]:


# Print the accuracy
print('Accuracy:', accuracy)


# In[18]:


# Save the trained model
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

