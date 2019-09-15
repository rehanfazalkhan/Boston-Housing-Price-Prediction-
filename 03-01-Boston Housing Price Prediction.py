#!/usr/bin/env python
# coding: utf-8

# # Regression
# 
# # Boston Housing Price Prediction 
# 

# In[1]:


import numpy as np
import pandas as pd
import sklearn


# In[2]:


print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
print(sklearn.__version__)


# In[4]:


df = pd.read_csv('data//housing.data', delim_whitespace=True, header=None)


# In[ ]:


df.head()


# | Code   | Description   |
# |:---|:---|
# |**CRIM** | per capita crime rate by town |
# |**ZN**  | proportion of residential land zoned for lots over 25,000 sq.ft. | 
# |**INDUS**  | proportion of non-retail business acres per town | 
# |**CHAS**  | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) | 
# |**NOX**  | nitric oxides concentration (parts per 10 million) | 
# |**RM**  | average number of rooms per dwelling | 
# |**AGE**  | proportion of owner-occupied units built prior to 1940 | 
# |**DIS**  | weighted distances to five Boston employment centres | 
# |**RAD**  | index of accessibility to radial highways | 
# |**TAX**  | full-value property-tax rate per $10,000 | 
# |**PTRATIO**  | pupil-teacher ratio by town | 
# |**B**  | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | 
# |**LSTAT**  | % lower status of the population | 
# |**MEDV**  | Median value of owner-occupied homes in \$1000's | 

# In[7]:


col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[8]:


df.columns = col_name


# In[23]:


df.head()


# # Exploratory Data Anaysis (EDA)

# In[24]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__)


# In[26]:


sns.pairplot(df, height=2.5)
plt.show()


# In[27]:


col_study = ['ZN', 'INDUS', 'NOX', 'RM']


# In[28]:


sns.pairplot(df[col_study], height=2.5)
plt.show()


# | Code   | Description   |
# |:---|:---|
# |**CRIM** | per capita crime rate by town |
# |**ZN**  | proportion of residential land zoned for lots over 25,000 sq.ft. | 
# |**INDUS**  | proportion of non-retail business acres per town | 
# |**CHAS**  | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) | 
# |**NOX**  | nitric oxides concentration (parts per 10 million) | 
# |**RM**  | average number of rooms per dwelling | 
# |**AGE**  | proportion of owner-occupied units built prior to 1940 | 
# |**DIS**  | weighted distances to five Boston employment centres | 
# |**RAD**  | index of accessibility to radial highways | 
# |**TAX**  | full-value property-tax rate per $10,000 | 
# |**PTRATIO**  | pupil-teacher ratio by town | 
# |**B**  | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | 
# |**LSTAT**  | % lower status of the population | 
# |**MEDV**  | Median value of owner-occupied homes in \$1000's | 

# In[31]:


col_study = ['PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[32]:


sns.pairplot(df[col_study], size=2.5);
plt.show()


# ***

# # Correlation Analysis and Feature Selection

# In[15]:


pd.options.display.float_format = '{:,.2f}'.format


# In[16]:


df.corr()


# In[17]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[18]:


plt.figure(figsize=(16,10))
sns.heatmap(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']].corr(), annot=True)
plt.show()


# ***

# # Linear Regression with Scikit-Learn

# In[6]:


df.head()


# In[14]:


X = df['RM'].values.reshape(-1,1)
print(X.shape)


# In[38]:


y = df['MEDV'].values


# In[36]:


from sklearn.linear_model import LinearRegression


# In[39]:


model = LinearRegression()


# In[40]:


model.fit(X, y)


# In[41]:


model.coef_


# In[42]:


model.intercept_


# In[27]:


plt.figure(figsize=(12,10));
sns.regplot(X, y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


# In[28]:


sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', size=10);
plt.show();


# ***

# In[29]:


X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
model.fit(X, y)
plt.figure(figsize=(12,10));
sns.regplot(X, y);
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


# In[30]:


sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='reg', size=10);
plt.show();


# ***

# # Robust Regression
# 
# Outlier Demo: [http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html](http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html)

# In[31]:


df.head()


# ## RANdom SAmple Consensus (RANSAC) Algorithm
# 
# link = [http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression](http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression)

# Each iteration performs the following steps:
# 
# 1. Select `min_samples` random samples from the original data and check whether the set of data is valid (see `is_data_valid`).
# 
# 2. Fit a model to the random subset (`base_estimator.fit`) and check whether the estimated model is valid (see `is_model_valid`).
# 
# 3. Classify all data as inliers or outliers by calculating the residuals to the estimated model (`base_estimator.predict(X) - y`) - all data samples with absolute residuals smaller than the `residual_threshold` are considered as inliers.
# 
# 4. Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model has the same number of inliers, it is only considered as the best model if it has better score.

# In[32]:


X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values


# In[33]:


from sklearn.linear_model import RANSACRegressor


# In[34]:


ransac = RANSACRegressor()


# In[35]:


ransac.fit(X, y)


# In[36]:


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[37]:


np.arange(3, 10, 1)


# In[38]:


line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))


# In[39]:


sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,10));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper left')
plt.show()


# In[40]:


ransac.estimator_.coef_


# In[41]:


ransac.estimator_.intercept_


# ***

# In[42]:


X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0, 40, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))


# In[43]:


sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,10));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper right')
plt.show()


# ***

# # Performance Evaluation of Regression Model

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


#X = df['LSTAT'].values.reshape(-1,1)
X = df.iloc[:, :-1].values


# In[46]:


y = df['MEDV'].values


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[48]:


lr = LinearRegression()


# In[49]:


lr.fit(X_train, y_train)


# In[50]:


y_train_pred = lr.predict(X_train)


# In[51]:


y_test_pred = lr.predict(X_test)


# ***

# # Method 1: Residual Analysis

# In[52]:


plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])
plt.show()


# ***

# # Method 2: Mean Squared Error (MSE)
# 
# $$MSE=\frac{1}{n}\sum^n_{i=1}(y_i-\hat{y}_i)^2$$
# 
# * The average value of the Sums of Squared Error cost function  
# 
# * Useful for comparing different regression models 
# 
# * For tuning parameters via a grid search and cross-validation

# In[53]:


from sklearn.metrics import mean_squared_error


# In[54]:


mean_squared_error(y_train, y_train_pred)


# In[55]:


mean_squared_error(y_test, y_test_pred)


# # Method 3: Coefficient of Determination, $R^2$
# 
# $$R^2 = 1 - \frac{SSE}{SST}$$
# 
# SSE: Sum of squared errors
# 
# SST: Total sum of squares

# In[6]:


from sklearn.metrics import r2_score


# In[57]:


r2_score(y_train, y_train_pred)


# In[58]:


r2_score(y_test, y_test_pred)


# ***

# # What does a Near Perfect Model Looks like?

# In[16]:


generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(1000)
y = 3 * x + np.random.randn(1000)
plt.figure(figsize = (10, 8))
plt.scatter(x, y);
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[14]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train.reshape(-1, 1), y_train)


y_train_pred = model.predict(X_train.reshape(-1, 1))
y_test_pred = model.predict(X_test.reshape(-1, 1))


# # Method 1: Residual Analysis

# In[63]:


plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])
plt.ylim([-25, 15])
plt.show()


# # Method 2: Mean Squared Error (MSE)

# In[64]:


mean_squared_error(y_train, y_train_pred)


# In[65]:


mean_squared_error(y_test, y_test_pred)


# # Method 3: Coefficient of Determination, $R^2$

# In[66]:


r2_score(y_train, y_train_pred)


# In[67]:


r2_score(y_test, y_test_pred)


# ***

# # Yet to be covered
# 
# * Mean Absolute Error
# * Stratified Shuffle Split
# * Feature Engineering. E.g., Combining Features, Designing derivative features
# * Data preparation. E.g., missing values etc.
# * Categorical features
# * Transform data / feature scaling. Scikit-learn Pipeline
# * Fine Tuning. E.g., Grid Search, Randomized Search
# * Ensemble Methods
# 

# ***
