#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('C:/Users/Yogesh Dhumal/Downloads/ecological_health_dataset.csv')
df.head()


# In[3]:


df.info()


# In[4]:


# Create dummy variables
df_dummies = pd.get_dummies(df['Pollution_Level'], prefix='Pollution_Level')
df_dummies = pd.get_dummies(df['Ecological_Health_Label'], prefix='Ecological_Health_Label')

# Concatenate the dummy variables with the original DataFrame
df = pd.concat([df, df_dummies], axis=1)


# In[5]:


df["Ecological_Health_Label"].unique()


# In[6]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the target variable
df['Ecological_Health_Label_encoded'] = label_encoder.fit_transform(df['Ecological_Health_Label'])



# In[7]:


df = df.drop(['Pollution_Level', 'Timestamp'], axis=1)


# In[8]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'df' with null values

# Handle null values (e.g., imputation)
df = df.fillna(df.mean())

# Create a correlation matrix
corr_matrix = df.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[9]:


#!pip install dataprep
from dataprep.eda import create_report
from dataprep.eda import create_report
# Create the EDA report
report = create_report(df)
create_report(df).show()
report.save('Ecological_Health _EDA_report_2.html')


# In[10]:


# Assuming `df` is your DataFrame and includes 'Ecological_Health_Label' and other numeric columns
correlation_matrix = df.corr()

# Extract correlation values for 'Ecological_Health_Label'
target_correlation = correlation_matrix['Ecological_Health_Label_encoded']

# Sort correlations in ascending order
sorted_correlation = target_correlation.sort_values()
print(sorted_correlation)


# In[11]:


df


# In[ ]:




