# Importing the required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing the data

df = pd.read_csv('bank-marketing.csv')
df.head()

# Renaming yes and no in different columns to dintinguish them

df['targeted'].replace(['yes','no'],['t_yes','t_no'], inplace = True)
df['default'].replace(['yes','no'],['d_yes','d_no'], inplace = True)
df['housing'].replace(['yes','no'],['h_yes','h_no'], inplace = True)
df['loan'].replace(['yes','no'],['l_yes','l_no'], inplace = True)

df['response'] = df.response.replace(['yes','no'],[1,0])

# Creating list of the best features after analysis
cols = ['education', 'salary', 'age', 'marital', 'default', 'job', 'balance', 'previous', 'housing','response', 'duration', 'campaign']

# Creating dataframe with required cols
df1 = df[cols]

# Encoding the data with 1 and 0 for Machine Learning Models

def ohe(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

df1 = ohe('job',df1)
df1 = ohe('marital',df1)
df1 = ohe('education',df1)
df1 = ohe('default',df1)
df1 = ohe('housing',df1)

# X for train and Y for Test

x = df1.drop('response', axis = 1)
y = df1.response

# Train test split
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 50)

# Scaling the data
scaler =  StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Random Forest Model

rf1 = RandomForestClassifier(n_estimators = 100,max_depth = 10, max_features = 'sqrt')
rf1.fit(X_train,y_train)
rf_predict = rf1.predict(X_test)

# Create a pickle file using steralization
import pickle
f = open("rf1.pkl","wb")
pickle.dump(rf1,f)
f.close()