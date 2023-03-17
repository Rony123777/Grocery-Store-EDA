#!/usr/bin/env python
# coding: utf-8

# #  Exploratory Data Analysis
# 
# # Problem Statement:
# Gala Groceries is a technology-led grocery store chain based in the USA. They rely heavily on new technologies, such as IoT to give them a competitive edge over other grocery stores. 
# 
# They pride themselves on providing the best quality, fresh produce from locally sourced suppliers. However, this comes with many challenges to consistently deliver on this objective year-round.
# 
#  Groceries are highly perishable items. If you overstock, you are wasting money on excessive storage and waste, but if you understock, then you risk losing customers. They want to know how to better stock the items that they sell.
# 
# This is a high-level business problem and will require you to dive into the data in order to formulate some questions and recommendations to the client about what else we need in order to answer that question.
# 
# Once you’re done with your analysis, we need you to summarize your findings and provide some suggestions as to what else we need in order to fulfill their business problem. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt1
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('sample_sales_data.csv')
df.head()


# In[3]:


df=df.iloc[:,1:]
df.head()


# In[4]:


df.info()


# # Descriptive statistics
# 

# transaction_id = this is a unique ID that is assigned to each transaction
# timestamp = this is the datetime at which the transaction was made
# product_id = this is an ID that is assigned to the product that was sold. Each product has a unique ID
# category = this is the category that the product is contained within
# customer_type = this is the type of customer that made the transaction
# unit_price = the price that 1 unit of this item sells for
# quantity = the number of units sold for this product within this transaction
# total = the total amount payable by the customer
# payment_type = the payment method used by the customer
# It is also interesting to look at the datatypes. We can see that there are 3 different datatypes within this dataset:
# 
# object = this column contains categorical values
# float64 = this column contains floating point numerical values (i.e. decimal numbers)
# int64 = this column contains integer values (whole numbers)
# Now let's compute some descriptive statistics of the numeric columns:

# In[5]:


df.describe()


# # Visualization

# To analyse the dataset, below are snippets of code that you can use as helper functions to visualise different columns within the dataset. They include:
# 
# 
# - plot_continuous_distribution = this is to visualise the distribution of numeric columns
# - get_unique_values = this is to show how many unique values are present within a column
# - plot_categorical_distribution = this is to visualise the distribution of categorical columns

# In[6]:


plt.figure(figsize=(15,5))
sns.displot(df['unit_price'],kde=True)
plt.title('Distribution Plot for Unit Price')


# This tell us that the distribution of `unit_price` is positively skewed, that is, there are more sales of products with a low unit_price compared to products with a high unit_price.
# 
# This makes sense, you would expect a grocery store to sell more products that are cheap, and just a few products that are really expensive.

# In[7]:


plt.figure(figsize=(20,5))
sns.displot(df['quantity'],kde=True)
plt.title('Distribution Plot for Quantity')


# The distribution of `quantity` is very different. We can see that only 4 unique values exist (1, 2, 3, and 4) and they are quite evenly distributed. It seems as though customers are buying in even quantities across 1 to 4 units

# In[8]:


plt.figure(figsize=(20,5))
sns.displot(df['total'],kde=True)
plt.title('Distribution Plot for Total')


# The `total` follows a similar distribution to `unit_price`. This you may expect, as the total is calculated as `unit_price x quantity`.
# 
# However, this distribution is even more positively skewed. Once again, using intuition, this distribution makes sense. You'd expect customers at a grocery store to generally make more transactions of low value and only occasionally make a transaction of a very high value.
# 
# Now let's turn our attention to the categorical columns within the dataset. 
# 
# Before visualising these columns, it is worth us understanding how many unique values these columns have. If a categorical column has 1000's of unique values, it will be very difficult to visualise.
# 

# In[9]:


def get_unique_values(data, column):
  num_unique_values = len(data[column].unique())
  value_counts = data[column].value_counts()
  print(f"Column: {column} has {num_unique_values} unique values\n")
  print(value_counts)


# In[10]:


get_unique_values(df, 'transaction_id')


# As explained previously, `transaction_id` is a unique ID column for each transaction. Since each row represents a unique transaction, this means that we have 7829 unique transaction IDs. Therefore, this column is not useful to visualise.

# In[11]:


get_unique_values(df, 'product_id')


# Similarly, `product_id` is an ID column, however it is unique based on the product that was sold within the transaction. From this computation, we can see that we have 300 unique product IDs, hence 300 unique products within the dataset. This is not worth visualising, but it certainly interesting to know. From the output of the helper function, we can see that the product most frequently was sold within this dataset was `ecac012c-1dec-41d4-9ebd-56fb7166f6d9`, sold 114 times during the week. Whereas the product least sold was `ec0bb9b5-45e3-4de8-963d-e92aa91a201e` sold just 3 times
# 

# In[12]:


get_unique_values(df, 'category')


# There are 22 unique values for `category`, with `fruit` and `vegetables` being the 2 most frequently purchased product categories and `spices and herbs` being the least. Let's visualise this too

# In[13]:


plt.figure(figsize=(20,5))
sns.countplot(df['category'])
plt.xticks(rotation=45)
plt.xlabel('category',fontsize=17)
plt.ylabel('Count',fontsize=17)


# In[14]:


get_unique_values(df, 'customer_type')


# There are 5 unique values for `customer_type`, and they seem to be evenly distributed. Let's visualise this:

# In[15]:


plt.figure(figsize=(10,6))
sns.countplot(df['customer_type'])
#plt.xticks(rotation=45)
plt.xlabel('Customer Type',fontsize=17)
plt.ylabel('Count',fontsize=17)


# From this sample of data, non-members appear to be the most frequent type of customers, closely followed by standard and premium customers

# In[16]:


get_unique_values(df, 'payment_type')


# There are 4 unique values for `payment_type`, and they seem to be quite evenly distributed once again. Let's visualise this:

# In[17]:


plt.figure(figsize=(10,6))
sns.countplot(df['payment_type'])
#plt.xticks(rotation=45)
plt.xlabel('Payment Type',fontsize=17)
plt.ylabel('Count',fontsize=17)


# Interestingly, cash seems to be the most frequently used method of payment from this sample of data, with debit cards being the least frequent. 
# 
# 
# This dataset is a sample from 1 store across 1 week. So it will be interesting to see if the population sample follows similar patterns.

# In[18]:


get_unique_values(df, 'timestamp')


# Clearly there are a lot of unique values for the timestamp column. 
# 
# However, you may have noticed something...
# 
# The column named `timestamp` appears to be categorical, but in actual fact it's not. This is a datetime, following the format of `2022-03-01 10:00:45 = YYYY-MM-DD HH:MM:SS`. Therefore, we must transform this column to reflect its true form.
# 
# A helper function is provided below to convert the column into a datetime column.

# In[19]:


def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy


# In[20]:


df = convert_to_datetime(df, 'timestamp')


# In[21]:


df.info()


# Using the `.info()` method again, we can see that the timestamp is now of type `datetime64[ns`, which indicates it is a datetime based data type. Now that this is a datetime column, we can explode this column out into its consitituent parts, e.g. we can explode datetime into `hour` for example.

# In[22]:


df['hour'] = df['timestamp'].dt.hour


# In[23]:


df.head()


# In[24]:


get_unique_values(df, 'hour')


# From this we can see that the 11th, 16th and 18th hour of the day are the top 3 hours of the day for transactions being processed. This is interesting, this would suggest that their busiest times of day may be just before lunch, and as people are on the way home from work. Once again, this is a small sample of data, so we can't make assumptions on the population sample of data, but it gives us insights to go back to the business with.

# ---
# 
# ##  Correlations
# 
# By now, you should have a good understanding of all the columns within the dataset, as well as the values that occur within each column. One more thing that we can do is to look at how each of the numerical columns are related to each other.
# 
# To do this, we can use `correlations`. Correlations measure how each numeric column is linearly related to each other. It is measured between -1 and 1. If a correlation between 2 columns is close to -1, it shows that there is a negative correlation, that is, as 1 increases, the other decreases. If a correlation between 2 columns is close to 1, it shows that they are positively correlated, that is, as 1 increases, so does the other. Therefore, correlations do not infer that one column causes the other, but it gives us an indication as to how the columns are linearly related.

# In[25]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# From this correlation matrix, we can see that the only columns that have a high correlation are `unit_price` and `total`. This is understandable because total is calculated used unit_price. 
# 
# All the other correlations are close to 0, indicating that there is not a significant positive or negative correlation between the numeric variables.
# 
# ---
# 
# ## Outcome - Summary
# 
# We have completed an initial exploratory data analysis on the sample of data provided. We should now have a solid understanding of the data. 
# 
# The client wants to know
# 
# ```
# "How to better stock the items that they sell"
# ```
# 
# From this dataset, it is impossible to answer that question. In order to make the next step on this project with the client, it is clear that:
# 
# - We need more rows of data. The current sample is only from 1 store and 1 week worth of data
# - We need to frame the specific problem statement that we want to solve. The current business problem is too broad, we should narrow down the focus in order to deliver a valuable end product
# - We need more features. Based on the problem statement that we move forward with, we need more columns (features) that may help us to understand the outcome that we're solving for
# 
# 

# In[ ]:




