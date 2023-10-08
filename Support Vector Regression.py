#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\Real estate.csv')
data


# In[3]:


data.describe()


# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()['Y house price of unit area']


# In[7]:


data=data.drop(['No', 'X1 transaction date', 'X2 house age'], axis=1)


# In[8]:


data.head()


# In[9]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data [[
    'X3 distance to the nearest MRT station', 
    'X4 number of convenience stores', 
#     'X5 latitude', 
#     'X6 longitude'
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
pd.options.display.float_format = '{:.2f}'.format
vif


# In[10]:


data.drop(['X5 latitude', 'X6 longitude'], axis=1, inplace=True)


# In[11]:


data.head()


# In[12]:


for i in data[['X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[13]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[14]:


for i in data[['X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])


# In[15]:


for i in data[['X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[16]:


data=data.reset_index(drop=True)


# In[17]:


data.head(2)


# In[18]:


X=data.drop('Y house price of unit area', axis=1)
y=data['Y house price of unit area']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[20]:


def evaluate(model, X_test, y_test):
    
    y_pred_test=model.predict(X_test)
    mae_test=metrics.mean_absolute_error(y_test, y_pred_test)
    mse_test=metrics.mean_squared_error(y_test, y_pred_test)
    rmse_test=np.sqrt(mse_test)
    r2_test=metrics.r2_score(y_test, y_pred_test)
    
    
    
    y_pred_train=model.predict(X_train)
    mae_train=metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train=metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=metrics.r2_score(y_train, y_pred_train)
    
    
    results_dict = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train*100],
        'Test': [mae_test, mse_test, rmse_test, r2_test*100]
    }

    results_df = pd.DataFrame(results_dict)
    
    print(results_df)


# In[21]:


base_model=SVR()
base_model.fit(X_train, y_train)
base_accuracy=evaluate(base_model, X_test, y_test)


# In[22]:


from sklearn.model_selection import RandomizedSearchCV

kernel = ['linear', 'poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto'] 

C = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]

epsilon = [0.1 , 0.01, 0.001, 0.0001]



random_grid = {'kernel': kernel,
               'gamma': gamma,
               'C': C,
               'epsilon': epsilon}
print(random_grid)


# In[23]:


svr_randomized = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 1, cv = 3, verbose=1, n_jobs = -1)

svr_randomized.fit(X_train, y_train)


# In[24]:


svr_randomized.best_params_


# In[25]:


optimized_model = SVR(kernel='rbf', gamma='auto', epsilon=0.001, C=1000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[26]:


optimized_model = SVR(kernel='rbf', gamma='scale', epsilon=0.001, C=1000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[27]:


optimized_model = SVR(kernel='rbf', gamma='auto', epsilon=0.1, C=100000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[28]:


optimized_model = SVR(kernel='rbf', gamma='auto', epsilon=0.001, C=10000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[29]:


optimized_model = SVR(kernel='rbf', gamma='auto', epsilon=0.00001, C=1000000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[30]:


optimized_model = SVR(kernel='rbf', gamma='auto', epsilon=0.1, C=1000)
optimized_model.fit(X_train, y_train) 
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[31]:


variables = []
train_r2_scores = []
test_r2_scores = []

for i in X_train.columns: 
    X_train_single = X_train[[i]]
    X_test_single = X_test[[i]]

    
    optimized_model.fit(X_train_single, y_train)
    
    
    y_pred_train_single = optimized_model.predict(X_train_single)
    train_r2 = metrics.r2_score(y_train, y_pred_train_single)
    
    

    y_pred_test_single = optimized_model.predict(X_test_single)
    test_r2 = metrics.r2_score(y_test, y_pred_test_single)

    variables.append(i)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    
    
    
results_df = pd.DataFrame({'Variable': variables, 'Train R2': train_r2_scores, 'Test R2': test_r2_scores})

results_df_sorted = results_df.sort_values(by='Test R2', ascending=False)

pd.options.display.float_format = '{:.4f}'.format

results_df_sorted


# In[ ]:




