#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AP Industries_Web_Scrapping.csv")
df.head()


# In[93]:


df['cap_Latitude'] = 17.6868
df['cap_Longitude'] = 83.2185


from haversine import haversine

loc1 = list(zip(df.Latitude, df.Longitude))
loc2 = list(zip(df.cap_Latitude, df.cap_Longitude))
df['distance_from_cap'] = ''
for ind in df.index: 
     df['distance_from_cap'].values[ind] = haversine(loc1[ind], loc2[ind])
#df.head()

list(df)


# In[94]:


df['Category'] = df['Category'].replace('Large.png"               ', 'Large')
df['Category'] = df['Category'].fillna("NA")
print(df['Category'].unique())

df.groupby('Category').size()

# Replacing with NA values with micro (selected based on mode)
df['Category'] = df['Category'].replace('NA', 'Micro')
print(df['Category'].unique())

print(df['District Name'].unique())

df.groupby('Sector Name').size()

# Pulp industry consists of packaging industry and other service providing shops, so replacing with service
df.loc[df['Activity Name'] == 'Pulp', 'Sector Name'] = "SERVICE"

# Plastic industry consists manufacturing of plastic products, so replacing with Engineering.
df.loc[df['Activity Name'] ==  'Plastics', 'Sector Name'] = "ENGINEERING"

print(df['Activity Name'].unique())

df['Activity Name'] = df['Activity Name'].replace([' Total Workers ', ' Industry as per pollution Index Category '], 'AUTOMOBILE SERVICING')
print((df['Activity Name'].unique()))

df['Activity Name'] = df['Activity Name'].replace('NA', 'AUTOMOBILE SERVICING')
print(df['Activity Name'].unique())

df.groupby('Activity Name').size()

print(df['Pollution Index Category'].unique())
df.groupby('Pollution Index Category').size()

df['Pollution Index Category'] = df['Pollution Index Category'].replace(' Total Workers ', 'NA')
df['Pollution Index Category'] = (df['Pollution Index Category'].fillna("Green"))
print(df['Pollution Index Category'].unique())

df.groupby('Pollution Index Category').size()
df['Pollution Index Category'] = df['Pollution Index Category'].replace('NA', 'Green')
print(df['Pollution Index Category'].unique())

df['Total Workers'].describe().round(1)

# replacing with nan values with median (2)
df['Total Workers'] = df['Total Workers'].fillna(2)
print(df['Total Workers'])

# Removing outliers
df.loc[df['distance_from_cap']> 1070, 'distance_from_cap'] = 1070
df['distance_from_cap'].describe().round(1)


# In[95]:


df[(df['Total Workers']> 9)].shape[0]


# In[96]:



# Removing outliers
df.loc[df['Total Workers']> 9, 'Total Workers'] = 9
df['Total Workers'].describe().round(1)


# In[56]:


df.isnull().sum()


# In[97]:


df['distance_from_cap']= pd.to_numeric(df['distance_from_cap'])
df['distance_from_cap'].describe().round(2)

df.info()


# In[84]:


df.describe()


# In[59]:


df.head()


# In[98]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Category']  = LE.fit_transform(df['Category'])
#df['Category'].dtype

df['District Name']  = LE.fit_transform(df['District Name'])                                    

df['Sector Name']  = LE.fit_transform(df['Sector Name'])

df['Pollution Index Category'] = LE.fit_transform(df['Pollution Index Category']).astype(float)
#df['Pollution Index Category'].dtype

df.head()


# In[99]:


#df1 = pd.get_dummies(data=df, columns=["Sector Name","Activity Name"], drop_first = True)
#df1.head()

#  removing Industry names and coordinates 
df1 = df.drop(['Unnamed: 0', 'Industry Name', 'Latitude', 'Longitude', 'cap_Latitude', 'cap_Longitude', 'Activity Name'], axis=1)
df1.head()


# In[10]:


# taking only 50 columns 
#plt.figure(figsize=(70,70))
sns.heatmap(df1.corr(), annot = True, cmap = 'coolwarm' ,fmt = '.0%')


# In[11]:


plt.boxplot(df1['distance_from_cap'])


# In[106]:


plt.boxplot(df1['Total Workers'])


# In[107]:


sns.distplot(df1['Total Workers'])


# In[108]:


sns.distplot(df1['distance_from_cap'])


# In[100]:


x = df1.drop([ 'Total Workers'], axis=1)

y = df1['Total Workers']
x.head()


# In[101]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    random_state=50)
X1_train.shape, X1_test.shape, y1_train.shape, y1_test.shape


# In[102]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

reg = LinearRegression()
reg = reg.fit(X1_train, y1_train)

lr_y_pred_train = reg.predict(X1_train)
lr_y_pred1_test = reg.predict(X1_test)

lr_mse_train = mean_squared_error(y1_train, lr_y_pred_train)
print("MSE", lr_mse_train)

lr_mse_test = mean_squared_error(y1_test, lr_y_pred1_test)
print("MSE", lr_mse_test)

lr_r2_train = r2_score(y1_train, lr_y_pred_train)
print("r2_score", lr_r2_train)

lr_r2_test = r2_score(y1_test, lr_y_pred1_test)
print("r2_score", lr_r2_test)


# In[103]:


rid_r = Ridge(alpha = 1, random_state = 0).fit(X1_train, y1_train)

y_pred_train_rid = rid_r.predict(X1_train)
y_pred_test_rid = rid_r.predict(X1_test)

rid_mse_train = mean_squared_error(y1_train, y_pred_train_rid)
print("Mse of RR on Test data",rid_mse_train.round(3))

rid_mse_test = mean_squared_error(y1_test, y_pred_test_rid)
print("Mse of RR on Test data",rid_mse_test.round(3))

rid_r2_train = r2_score(y1_train, y_pred_train_rid)
print("R2_score of RR on Test data", rid_r2_train.round(3))

rid_r2_test = r2_score(y1_test, y_pred_test_rid)
print("R2_score of RR on Test data", rid_r2_test.round(3))


# In[14]:


from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(random_state = 0)

rfr = rf1.fit(X1_train, y1_train)

rf_Y_pred_test = rfr.predict(X1_test)

rf_mse_test = mean_squared_error(y1_test, rf_Y_pred_test)
print("Mse of RFR on Test data", rf_mse_test.round(3))

rf_r2_test = r2_score(y1_test, rf_Y_pred_test)
print("R2_score of RFR on Test data", rf_r2_test.round(3))


# In[15]:


error = y1_test - rf_Y_pred_test


# In[16]:


# ploting Q-Q plot
from scipy import stats
stats.probplot(error , dist = 'norm', plot=plt)
 


# In[116]:


plt.scatter(y = rf_Y_pred_test, x = y1_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# In[117]:


plt.scatter(y = error, x = y1_test)
plt.xlabel('Input Feature')
plt.ylabel('Residuals')
plt.show()


# In[ ]:





# In[20]:


max_depth = [int(x) for x in np.linspace(start = 1, stop = 60, num = 60)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

tree_para = {
            'criterion':['mse'],
            'max_features' : ['auto'],
            'max_depth' : max_depth,
            'min_samples_split' : min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'random_state' : random_state
            }


clf = GridSearchCV(dt, tree_para, cv=10)
clf.fit(X1_train, y1_train)

print(clf.best_score_)
print(clf.best_params_)


# In[23]:


dt1 = DecisionTreeRegressor(
                                criterion = 'mse',
                                min_samples_leaf = 10,
                                max_features = 'auto',
                                max_depth = 11,
                                random_state = 5
                                )

dt1.fit(X1_train, y1_train)

dt_y_pred_train = dt1.predict(X1_train)
dt_y_pred_dt_test = dt1.predict(X1_test)

dt_mse_train = mean_squared_error(y1_train, dt_y_pred_train)
print("Mse of DT on Train data", dt_mse_train.round(3))

dt_mse_test = mean_squared_error(y1_test, dt_y_pred_dt_test)
print("Mse of DT on Test data", dt_mse_test.round(3))

dt_r2_train = r2_score(y1_train, dt_y_pred_train)
print("R2_score of DT on Train data", dt_r2_train.round(3))

dt_r2_test = r2_score(y1_test, dt_y_pred_dt_test)
print("R2_score of DT on Test data", dt_r2_test.round(3))


# In[27]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto']

max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'random_state' : random_state
               }

print(random_grid)

rf = RandomizedSearchCV(estimator= RandomForestRegressor(), param_distributions = random_grid, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
rf

rf.fit(X1_train, y1_train)

print("Best score", rf.best_score_)
print("Best parameters", rf.best_params_)


# In[28]:


from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(
                             n_estimators = 700, 
                             min_samples_split = 10,
                             min_samples_leaf = 2,
                             max_features = 'auto',
                             max_depth = 11,
                             random_state = 5
                            )

rfr = rf1.fit(X1_train, y1_train)

rf_Y_pred_train = rfr.predict(X1_train)

rf_Y_pred_test = rfr.predict(X1_test)

rf_mse_train = mean_squared_error(y1_train, rf_Y_pred_train)
print("Mse of RFR on Trained data", rf_mse_train.round(3))

rf_mse_test = mean_squared_error(y1_test, rf_Y_pred_test)
print("Mse of RFR on Test data", rf_mse_test.round(3))

rf_r2_train = r2_score(y1_train, rf_Y_pred_train)
print("R2_score of RFR on Trained data", rf_r2_train.round(3))

rf_r2_test = r2_score(y1_test, rf_Y_pred_test)
print("R2_score of RFR on Test data", rf_r2_test.round(3))


# In[40]:


from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor()

base_estimator = [dt1]

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_samples = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]

max_features = ['auto']

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid = {
                'base_estimator': base_estimator,
                'n_estimators' : n_estimators,
                'max_samples' : max_samples,
                'random_state' : random_state
               }

print(random_grid)

rf = RandomizedSearchCV(estimator= BaggingRegressor(), param_distributions = random_grid, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
rf

rf.fit(X1_train, y1_train)

print("Best score", rf.best_score_)
print("Best parameters", rf.best_params_)


# In[43]:


from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(
                        base_estimator = dt1, 
                        max_samples=2, 
                        n_estimators=900, 
                        random_state = 1
                      )

bag.fit(X1_train, y1_train)
#bag.score(X1_test, y1_test).round(2)

y_pred_train_bag = bag.predict(X1_train)
y_pred_test_bag = bag.predict(X1_test)

bag_mse_train = mean_squared_error(y1_train, y_pred_train_bag)
print("MSE of bagging on Training data", bag_mse_train)

bag_mse_test = mean_squared_error(y1_test, y_pred_test_bag)
print("MSE of bagging on Test data", bag_mse_test)

bag_r2_train = r2_score(y1_train, y_pred_train_bag)
print("R2_score of bagging on Train data", bag_r2_train.round(3))

bag_r2_test = r2_score(y1_test, y_pred_test_bag)
print("R2_score of bagging on Test data", bag_r2_test.round(3))


# In[44]:


# Without Base estimator
bag1 = BaggingRegressor(
                        max_samples=2, 
                        n_estimators=900, 
                        random_state = 1
                      )

bag1.fit(X1_train, y1_train)
#bag.score(X1_test, y1_test).round(2)

y_pred_train_bag1 = bag.predict(X1_train)
y_pred_test_bag1 = bag.predict(X1_test)

bag_mse_train1 = mean_squared_error(y1_train, y_pred_train_bag1)
print("MSE of bagging on Training data", bag_mse_train1)

bag_mse_test1 = mean_squared_error(y1_test, y_pred_test_bag1)
print("MSE of bagging on Test data", bag_mse_test1)

bag_r2_train1 = r2_score(y1_train, y_pred_train_bag1)
print("R2_score of bagging on Train data", bag_r2_train1.round(3))

bag_r2_test1 = r2_score(y1_test, y_pred_test_bag1)
print("R2_score of bagging on Test data", bag_r2_test1.round(3))


# In[47]:


from sklearn.linear_model import SGDRegressor

alpha = np.arange(0.001, 0.015,0.001)
sgd_rmse_train = []
sgd_rmse_test = []
sgd_r2_train = []
sgd_r2_test = []

for x in alpha:
    sgd = SGDRegressor(learning_rate='constant', eta0=x)
    sgd.fit(X1_train,y1_train)
    sgd_y_pred_train = sgd.predict(X1_train)
    sgd_y_pred_test = sgd.predict(X1_test)
    sgd_rmse_train.append(mean_squared_error(y1_train, sgd_y_pred_train))
    sgd_rmse_test.append(mean_squared_error(y1_test, sgd_y_pred_test))
    sgd_r2_train.append(r2_score(y1_train, sgd_y_pred_train))
    sgd_r2_test.append(r2_score(y1_test, sgd_y_pred_test))

print(sgd_rmse_train)
print(sgd_rmse_test)
print(sgd_r2_train)
print(sgd_r2_test)

err_train = pd.DataFrame(sgd_rmse_train)

err_test = pd.DataFrame(sgd_rmse_test).round(3)

acc_train = pd.DataFrame(sgd_r2_train).round(3)

acc_test = pd.DataFrame(sgd_r2_test).round(3)


# In[48]:


plt.figure(figsize=(15,6))
plt.plot(alpha, err_test)


# In[49]:


plt.plot(alpha, acc_test)


# In[73]:


sgd = SGDRegressor(alpha = 0.01, learning_rate='constant', eta0=0.001)
sgd.fit(X1_train, y1_train)

sgd_y_pred_train1 = sgd.predict(X1_train)

sgd_y_pred_test1 = sgd.predict(X1_test)

sgd_mse_train1 = mean_squared_error(y1_train, sgd_y_pred_train1)
print("MSE of XGB on Training data", sgd_mse_train1)

sgd_mse_test1 = mean_squared_error(y1_test, sgd_y_pred_test1)
print("MSE of XGB on Test data", sgd_mse_test1)

sgd_r2_train1 = r2_score(y1_train, sgd_y_pred_train1)
print("R2_score of XGB on Train data", sgd_r2_train1.round(3))


sgd_r2_test1 = r2_score(y1_test, sgd_y_pred_test1)
print("R2_score of XGB on Test data", sgd_r2_test1.round(3))


# In[46]:


import xgboost as xg
xgreg = xg.XGBRegressor()
xgreg.fit(X1_train, y1_train)

xgreg_y_pred_train = xgreg.predict(X1_train)

xgreg_y_pred_test = xgreg.predict(X1_test)

xgreg_mse_train = mean_squared_error(y1_train, xgreg_y_pred_train)
print("MSE of XGB on Training data", xgreg_mse_train)

xgreg_mse_test = mean_squared_error(y1_test, xgreg_y_pred_test)
print("MSE of XGB on Test data", xgreg_mse_test)

xgreg_r2_train = r2_score(y1_train, xgreg_y_pred_train)
print("R2_score of XGB on Train data", xgreg_r2_train.round(3))

xgreg_r2_test = r2_score(y1_test, xgreg_y_pred_test)
print("R2_score of XGB on Test data", xgreg_r2_test.round(3))


# In[ ]:





# In[ ]:





# In[ ]:




