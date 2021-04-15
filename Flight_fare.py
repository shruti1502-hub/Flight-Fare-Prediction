#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib and seaborn help in data visualisation
sns.set()
#data is in excel form so has to be read using read_Excel


# In[3]:


train_data = pd.read_excel(r"Data_Train.xlsx")
pd.set_option('display.max_columns',None)


# In[4]:


train_data.head() #top 5 rows


# In[5]:


train_data.info()


# In[6]:


train_data["Duration"].value_counts()


# In[7]:


train_data.shape


# In[8]:


train_data.dropna(inplace = True) #dropping NaN values


# In[9]:


train_data.shape


# In[10]:


train_data.isnull().sum()


# EDA-Exploratory Data Analysis
# 
# dep_time is in string format so we need to take out hours and minutes separately 
# same for duration
# 
# pandas to_datetime converts object data type to datetime type
# 
# .dt.day method will extract only day of that date
# .dt.month method will extract only month of that date
# 

# In[11]:


train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format= "%d/%m/%Y").dt.day


# In[12]:


train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[13]:


train_data.head()


# In[14]:


#since we have converted Date_of_Journey column into integers,Now we can drop it off
 
train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[15]:


train_data.head()


# In[16]:


#Similar to Date_of_Journey we can extract values from Dep_Time

#Extracting Hours
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

#Extracting Minutes
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

#Now we can drop off Dep_Time as it is of no use
train_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[17]:


train_data.head()


# In[18]:


#Similar to Date_of_Journey we can extract values from arrival_time

#Extracting hours
train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

#Extracting minutes
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

#Drooping Arrival_Time as it is of no use
train_data.drop(["Arrival_Time"], axis=1, inplace = True)


# In[19]:


train_data.head()


# In[20]:


"2h 50m".split()
len("2h 50m".split())


# In[21]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[22]:


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins


# In[23]:


train_data.head()


# In[24]:


train_data.drop(["Duration"], axis=1, inplace = True)


# In[25]:


train_data.head()


# Handling Categorical Data
# 
# 1. Nominal Data-> data are not in any order --> OneHotEncoder used in this case
# 2. Ordinal Data-> data are in order--> LabelEncoder is used in this case

# In[26]:


train_data["Airline"].value_counts()


# In[27]:


#From graph we can see jet Airways has the highest price
#Apart from first airline all have similar median

#PLotting a graph of airline vs price
sns.catplot(y = "Price" , x = "Airline" , data = train_data.sort_values("Price" , ascending = False), kind="boxen", height = 6, aspect = 3)


# As airline is Nominal categorical data becaase all airlines provide same services, no airline is greater than other
# so we will perform OneHotEncoding
# 

# In[28]:


Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline,drop_first = True) #drop first is used to drop tr first column
#get_dummies is OneHotEncoding technique
Airline.head()


# In[29]:


train_data["Source"].value_counts()


# In[30]:


#Source vs Price

sns.catplot(y = "Price" , x = "Source" , data = train_data.sort_values("Price", ascending = False), kind = "boxen", height = 4 , aspect = 3)
plt.show()


# In[31]:


#Source is Nominal Categorical we will perfom OneHotEncoding

Source = train_data[["Source"]]

Source = pd.get_dummies(Source,drop_first= True)

Source.head()


# In[32]:


train_data["Destination"].value_counts()


# In[33]:


#As destination is Nominal Categorical we will perform Onehotencoding

Destination =  train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first  = True)

Destination.head()


# In[34]:


train_data["Route"]


# In[35]:


#Additional_Info contains almost 80% no_info
#Route and total_stops are related to each other

train_data.drop(["Route", "Additional_Info"] , axis = 1, inplace = True)


# In[36]:


train_data.head()


# In[37]:


train_data["Total_Stops"].value_counts()


# In[38]:


#As this is the case of Ordinal Categorical Data type we perform LabelEncodr 
#hEre values are assigned with corresponding keys

train_data.replace({"non-stop":0, "1 stop": 1, "2 stops":2, "3 stops": 3,"4 stops":4}, inplace = True)


# In[39]:


train_data.head()


# In[40]:


#Concatenate dataframe --> train_data + Airline + Source + destination

data_train = pd.concat([train_data,Airline,Source,Destination], axis = 1)


# In[41]:


data_train.head()


# In[42]:


data_train.drop(["Airline","Source","Destination"], axis = 1,inplace = True)


# In[43]:


train_data.head()


# In[44]:


data_train.shape


# # Test Set
# We cannot do pre-processing of test and train data together due to data leakage
# DataLeakage- leads to overfitting and sharing of data features
# 

# In[45]:


test_data  = pd.read_excel(r"Test_set.xlsx")


# In[46]:


test_data.head()


# In[47]:


#Preprocessing

print('Test Data Info')
print("-"*75)
print(test_data.info())

print()
print()

print("Null Values :")
print("-"*75)
test_data.dropna(inplace=True)
print(test_data.isnull().sum())

#Explratory Data Analysis

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[48]:


#Duration

duration= list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() +" 0m" #Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]        #Adds 0 hour
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0])) #Extract hours from duration
     
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1])) #Extracts only minutes from duration
    


# In[49]:


#Adding duratio  column to dataset
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace= True)


# In[50]:


#Categorical Data
print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first = True)

print()

print("Source")
print("-"*75)
print(test_data["Airline"].value_counts())
Source = pd.get_dummies(test_data["Airline"],drop_first = True)

print()

print("Destination")
print("-"*75)
print(test_data["Source"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)


# In[51]:


#Additional info contains 80% no info
#Route and total_stops are relate to each othet
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

#Replacing Total_Stops
test_data.replace({"non-stop":0 , "1 stop": 1, "2 stops":2 , "3 stops":3 , "4 stops":4}, inplace = True)

#Concatenate Dataframe--> test_data+Airline+Source+Destination

data_test = pd.concat([test_data,Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[52]:


data_test.head()


# # Feature Selection
#  Finding out best feature whuch will contribute and have good relation with target variable. Feature selection mthods are
#  
# 1. heatmap
# 2. feature_importance
# 3. SelectKBest
# 

# In[53]:


data_train.shape


# In[54]:


data_train.columns


# In[55]:


X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[56]:


y=data_train.iloc[:,1]


# In[57]:


y.head()


# In[58]:


#Find correlation between indepedent and depemdent variables
 
plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(),annot = True, cmap = "RdYlGn")

plt.show()


# In[59]:


#Important feature using ExtraTreesRegressor

from sklearn.ensemble import  ExtraTreesRegressor
selection= ExtraTreesRegressor()
selection.fit(X,y)


# In[60]:


print(selection.feature_importances_)


# In[61]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # Fitting model using Random Forest
# 
# 1. Split dataset into train and test set in order to prediction w.r.t X_test
# 2. If needed do scaling of data
# - Scaling is not done in random forest
# 3. Import model
# 4. Fit the data
# 5. Predict w.r.t X_test
# 6. In regression check RSME Score
# 7. PLot graph

# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42)


# In[63]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[64]:


y_pred = reg_rf.predict(X_test)


# In[65]:


reg_rf.score(X_train,y_train)


# In[66]:


reg_rf.score(X_test,y_test)


# In[72]:


sns.distplot(y_test-y_pred)
plt.show()


# In[73]:


plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[74]:


from sklearn import metrics


# In[76]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[77]:


# RMSE/ (max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[78]:


metrics.r2_score(y_test,y_pred)


# # Hyperparameter Tuning
# 
# choose following method for hyperparameter tuning
#    1. RandomizedSearchCV-->Fast
#    2. GridSearchCV
#    
#    Assign hyperparameter in form of dictionery
#    fit the model
#    check best parameters and best score

# In[79]:


from sklearn.model_selection import RandomizedSearchCV


# In[81]:


#Randomized Search CV

#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#Number of features to consider at every split
max_features = ['auto','sqrt']
#Max no of levels in a tree
max_depth = [int(x) for x in np.linspace(5,30, num = 6)]
#Min no of samples required to split a nodde
min_samples_split = [2,5,10,15,100]
#min no of samples required at each node
min_samples_leaf = [1,2,5,10]


# In[84]:


#create the random grid

random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}


# In[86]:


#Randomsearch of parameters, using 5 fold cross validdation, search across 100diff combinations
rf_random= RandomizedSearchCV(estimator = reg_rf,param_distributions = random_grid,
                             scoring = 'neg_mean_squared_error', n_iter = 10, cv= 5, verbose = 2, random_state = 42, n_jobs = 1)


# In[87]:


rf_random.fit(X_train, y_train)


# In[88]:


rf_random.best_params_


# In[90]:


prediction = rf_random.predict(X_test)


# In[91]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[92]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[ ]:




