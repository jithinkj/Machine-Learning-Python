
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor # I'm using decision tree model for this problem
from sklearn.metrics import mean_absolute_error #MAE 


# In[14]:


#Load all csv file 
sample_submission_file=pd.read_csv(r"C:\Users\jithin\Desktop\Machine Learning Python\sample_submission.csv")
test_file=pd.read_csv(r"C:\Users\jithin\Desktop\Machine Learning Python\test.csv")
train_file=pd.read_csv(r"C:\Users\jithin\Desktop\Machine Learning Python\train.csv")


# In[16]:


#train_file.head()
train_file.columns


# In[21]:


#making Sale price as target variable
train_y=train_file.SalePrice
#test_y=test_file.SalePrice
#Feature engineering, selecting some features to practice how decision model works
features=['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X=train_file[features]
test_X=test_file[features]
#test_file.columns


# In[22]:


#Decision Tree model
model=DecisionTreeRegressor(random_state=1)
model.fit(train_X,train_y)


# In[25]:


#saving prediction values
test_predictions=model.predict(test_X)
test_predictions


# In[29]:


#making a dataframe with predicted sales price and test id
output=pd.DataFrame({'Id':test_file.Id,'SalePrice':test_predictions})


# In[31]:


output.head()


# In[33]:


#saving to a csv file for submission to kaggle
output.to_csv('submission.csv',index=False)

