#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import lime
import lime.lime_tabular
import shap
import numpy as np


# In[13]:


# Gathering and preprocessing the data
# Load the dataset
iris_data = datasets.load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['class'] = pd.Series(iris_data.target)


# In[15]:


# Preprocess the data
# Convert categorical variable to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["class"])


# In[19]:


# Building and training the model
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("petal length (cm)", axis=1), df["petal length (cm)"], test_size=0.2, random_state=42)


# In[21]:


# Creating a linear regerssion model
regressor = LinearRegression()


# In[22]:


# Train the model
regressor.fit(X_train, y_train)


# In[23]:


# Evaluate the model
score = regressor.score(X_test, y_test)
print("R^2 score:", score)


# In[24]:


# Generating explanations of the model's predictions
# Select a sample from the testing set
idx = np.random.randint(len(X_test))
sample = X_test.iloc[idx]
actual_petal_length = y_test.iloc[idx]


# In[25]:


# Generate an explanation of why the model predicted a certain petal length for that sample using LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=["petal length (cm)"], verbose=True, mode="regression")
exp = explainer.explain_instance(sample.values, regressor.predict, num_features=len(X_train.columns))


# In[26]:


# Print the LIME explanation
print("LIME Explanation:")
print(exp.as_list())


# In[27]:


# Generate a global explanation of the model's predictions using SHAP
explainer = shap.Explainer(regressor, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

