#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from pandasql import sqldf
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


# Loeading the dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)


# In[3]:


# Explorting the DataFrame 
df.head(n=5)


# In[4]:


# Checking DataFrame column types 
df.info()


# In[5]:


# Percentage of NaN records in the DataFrame 
def nan_check(df):
    nan_count = df.isnull().sum()
    nan_percentage = df.isna().sum() / len(df) * 100
    return nan_count, nan_percentage

# Calling the function and printing the results
nan_percentages = nan_check(df)
print(nan_percentages)


# In[6]:


# Checking min/max of CustomerIDs 
customer_id_min = df['CustomerID'].min()
customer_id_max = df['CustomerID'].max()

print("Minimum CustomerID is:", customer_id_min,"Maximum CustomerID is:", customer_id_max)


# In[7]:


# Function to generate a random number above 18287
def generate_random_id():
    return random.randint(18288, 99999)

# Applying the function to NaN values in the 'CustomerID' column
df['CustomerID'] = df['CustomerID'].apply(lambda x: generate_random_id() if pd.isna(x) else x)

# Changing the datatype to int64
df['CustomerID'] = df['CustomerID'].astype('int64')


# In[8]:


def df_preprocessing(df):
    # Dropping rows from DataFrame 'df' where 'Description' column has NaN values
    # Making a copy of the DataFrame to avoid SettingWithCopyWarning
    df_processed = df.dropna(subset=['Description']).copy()

    # Converting 'InvoiceDate' to datetime format
    df_processed['InvoiceDate'] = pd.to_datetime(df_processed['InvoiceDate'])

    # Extracting year, month, and day from 'InvoiceDate' and creating new columns 'YearKey', 'MonthKey', and 'DayKey'
    df_processed['YearKey'] = df_processed['InvoiceDate'].dt.year
    df_processed['MonthKey'] = df_processed['InvoiceDate'].dt.month
    df_processed['DayKey'] = df_processed['InvoiceDate'].dt.day

    # Creating Total Amount column
    df_processed["TotalAmount"] = df_processed["Quantity"] * df_processed["UnitPrice"]

    return df_processed

# Applying the function to your DataFrame
df = df_preprocessing(df)


# In[9]:


# Explorting the DataFrame 
df.head(n=5)


# In[10]:


# Checking unique countries 
df["Country"].unique()


# In[11]:


# Data Validation - Checking Percentage of NaN records in the DataFrame 
def nan_check(df):
    nan_count = df.isnull().sum()
    nan_percentage = df.isna().sum() / len(df) * 100
    return nan_count, nan_percentage

# Calling the function and printing the results
nan_percentages = nan_check(df)
print(nan_percentages)


# In[12]:


# RFM DataFrame generation function 
def rfm(df):

    snapshot_date = max(df['InvoiceDate']) + pd.DateOffset(days=1)  # Calculating recency score

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
    })

    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'MonetaryValue'}, inplace=True)

    return rfm 


df_rfm = rfm(df)


# In[13]:


# Resetting the index of the df_rfm DataFrame if 'CustomerID' is set as the index
df_rfm = df_rfm.reset_index()

# Calculating the customer base size by counting distinct CustomerIDs
customer_base_size = df_rfm['CustomerID'].nunique()

customer_base_size


# In[14]:


# Checking RFM DataFrame 
df_rfm.head(n=5)


# In[15]:


# Checking descriptive statistics for the RFM features
df_rfm.describe()


# In[16]:


# Creating RFM bins based on quartiles - 1 to 4 range for Recency, Frequency and Monetary Values
def rfm_scores(df_rfm):
    # Calculating quartile thresholds for the 'Recency', 'Frequency', and 'MonetaryValue' columns
    quartiles_recency = df_rfm['Recency'].quantile([0.25, 0.5, 0.75, 1.0])
    quartiles_frequency = df_rfm['Frequency'].quantile([0.25, 0.5, 0.75, 1.0])
    quartiles_monetary = df_rfm['MonetaryValue'].quantile([0.25, 0.5, 0.75, 1.0])

    # Creating bins for recency, frequency, and monetary values using float('inf') for infinity
    recency_bins = [-float('inf'), quartiles_recency[0.25], quartiles_recency[0.50], quartiles_recency[0.75], float('inf')]
    frequency_bins = [-float('inf'), quartiles_frequency[0.25], quartiles_frequency[0.50], quartiles_frequency[0.75], float('inf')]
    monetary_bins = [-float('inf'), quartiles_monetary[0.25], quartiles_monetary[0.50], quartiles_monetary[0.75], float('inf')]

    # Assigning scores based on the bins
    df_rfm['R_Score'] = pd.cut(df_rfm['Recency'], bins=recency_bins, labels=[1, 2, 3, 4], include_lowest=True)
    df_rfm['F_Score'] = pd.cut(df_rfm['Frequency'], bins=frequency_bins, labels=[1, 2, 3, 4], include_lowest=True)
    df_rfm['M_Score'] = pd.cut(df_rfm['MonetaryValue'], bins=monetary_bins, labels=[1, 2, 3, 4], include_lowest=True)

    return df_rfm

# Applying the function to your DataFrame
df_rfm = rfm_scores(df_rfm)


# In[17]:


# Checking new DataFrame with RFM bins 
df_rfm.head(n=5)


# In[18]:


# Checking quartiles - Recency (R)
df_rfm['Recency'].quantile([0.25, 0.5, 0.75, 1.0])


# In[19]:


# Checking quartiles - Frequency (F)
df_rfm['Frequency'].quantile([0.25, 0.5, 0.75, 1.0])


# In[20]:


# Checking quartiles - MonetaryValue (M)
df_rfm['MonetaryValue'].quantile([0.25, 0.5, 0.75, 1.0])


# In[21]:


# Creating a histogram using seaborn to visualize the distribution of 'Recency'
# If histplot is not available, using distplot
sns.distplot(df_rfm['Recency'], bins=30, kde=False)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Distribution of Recency in RFM DataFrame')

plt.show()


# In[22]:


# Storing the RFM features in a list for K-Means

X = df_rfm[['R_Score', 'F_Score', 'M_Score']]


# In[23]:


X 


# In[24]:


# Calculate inertia (sum of squared distances) for different values of k
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init= 10, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)


# In[25]:


# Plot the elbow curve
plt.figure(figsize=(8, 6),dpi=150)
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for K-means Clustering')
plt.grid(True)
plt.show()

# Optimal number is clusters is k=4 


# In[26]:


# Perform K-means clustering with best K
clusters_kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
df_rfm['Cluster'] = clusters_kmeans.fit_predict(X)


# In[27]:


df_rfm


# In[28]:


# Assuming 'Cluster', 'R_Score', 'F_Score', and 'M_Score' are columns in df_rfm
# Convert categorical scores to numerical if they are not already
df_rfm['R_Score'] = df_rfm['R_Score'].astype(int)
df_rfm['F_Score'] = df_rfm['F_Score'].astype(int)
df_rfm['M_Score'] = df_rfm['M_Score'].astype(int)

# Group by cluster and calculate mean values
cluster_summary = df_rfm.groupby('Cluster').agg({
    'R_Score': 'mean',
    'F_Score': 'mean',
    'M_Score': 'mean'
}).reset_index()

cluster_summary


# In[29]:


import plotly.express as px

# Creating a 3D scatter plot using Plotly
fig = px.scatter_3d(df_rfm, x='Recency', y='Frequency', z='MonetaryValue',
                    color='Cluster', labels={'Cluster': 'Cluster #'},
                    title='3D Plot of RFM Clusters')

# Customize the layout
fig.update_layout(
    width=1000,  # Customize the width of the plot (in pixels)
    height=600,  # Customize the height of the plot (in pixels)
    legend_title_text='Cluster #'
)

# Show the plot
fig.show()


# In[33]:


# Calculating the total number of customers for percentage calculation
total_customers = df_rfm['CustomerID'].nunique()

# Grouping df_rfm by 'Cluster' and counting distinct CustomerIDs in each cluster
cluster_customer_count = df_rfm.groupby('Cluster')['CustomerID'].nunique().reset_index()
cluster_customer_count.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)

# Adding a column for percentages in the cluster_customer_count DataFrame
cluster_customer_count['Percentage'] = (cluster_customer_count['CustomerCount'] / total_customers) * 100

cluster_customer_count

