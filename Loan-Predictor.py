#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# In[2]:


#Reading and assigning csv files to variables
df_x=pd.read_csv('loan_train.csv')
df_y=pd.read_csv('loan_test.csv')


# In[3]:


df_y


# In[4]:


len(df_x)


# In[5]:


len(df_x.columns)


# In[6]:


df_x.shape


# In[7]:


df_x_columns = df_x.columns
df_x_columns


# In[8]:


df_x.describe()


# In[9]:


df_x.info()


# In[10]:


df_y.info()


# In[11]:


#calclulating null values
df_x.isna().sum()


# ## Cleaning of data

# In[12]:


# filling null values with the smaller number from values count
df_x['Gender'].value_counts()
df_x['Gender'].fillna('Female',inplace=True)
df_x.isna().sum()


# In[13]:


# filling null values with the smaller number from values count
df_x['Married'].value_counts()
df_x['Married'].fillna('No',inplace=True)
df_x.isna().sum()


# In[14]:


# filling null values with the smaller number from values count
df_x['Dependents'].value_counts()
df_x['Dependents'].fillna('3+',inplace=True)
df_x.isna().sum()


# In[15]:


# filling null values with the smaller number from values count
df_x['Self_Employed'].value_counts()
df_x['Self_Employed'].fillna('Yes',inplace=True)
df_x.isna().sum()


# In[16]:


# filling null values with the mean of LoanAmount column as it doesn't have discrete values
df_x['LoanAmount'].value_counts()
df_x['LoanAmount'].fillna(df_x['LoanAmount'].mean(),inplace=True)
df_x.isna().sum()


# In[17]:


# filling null values with the larger number from values count
df_x['Loan_Amount_Term'].value_counts()
df_x['Loan_Amount_Term'].fillna('360.0',inplace=True)
df_x.isna().sum()


# In[18]:


# filling null values with the smaller number from values count
df_x['Credit_History'].value_counts()
df_x['Credit_History'].fillna('0.0',inplace=True)
df_x.isna().sum()


# In[19]:


sns.heatmap(df_x.isnull())


# ## Cleaning of testing data

# In[20]:


df_y.isna().sum()
df_y['LoanAmount'].value_counts()


# In[21]:


#filling null values in all testing data columns
cols=['Gender','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']
for x in range(len(cols)):
    if cols[x]=='Gender':
        df_y['Gender'].fillna('Female',inplace=True)
    elif cols[x]=='Dependents':
        df_y['Dependents'].fillna('2',inplace=True)
    elif cols[x]=='Self_Employed':
        df_y['Self_Employed'].fillna('Yes',inplace=True)
    elif cols[x]=='LoanAmount':
        df_y['LoanAmount'].fillna(150.0,inplace=True)
    elif cols[x]=='Loan_Amount_Term':
        df_y['Loan_Amount_Term'].fillna(180.0,inplace=True)
    else:
        df_y['Credit_History'].fillna(0.0,inplace=True)


# In[22]:


df_y['Credit_History'].value_counts()


# In[23]:


df_y.isna().sum()


# In[24]:


sns.heatmap(df_y.isnull())


# # Exploratory Data Analysis

# In[25]:


#Checking loan status for the applicants
plt.style.use('ggplot')
df_x['Loan_Status'].value_counts().plot.bar(title='Loan Status',rot=0)
print(df_x['Loan_Status'].value_counts())


# In[26]:


#Checking Other categorical variables
#1.Gender variable
plt.style.use('ggplot')
df_x['Gender'].value_counts().plot.bar(title='Gender',rot=0)
print(df_x['Gender'].value_counts())


# In[27]:


#2.Marrital status
plt.style.use('ggplot')
df_x['Married'].value_counts().plot.bar(title='Married',rot=0)
print(df_x['Married'].value_counts())


# In[28]:


#3. self employed
plt.style.use('ggplot')
df_x['Self_Employed'].value_counts().plot.bar(title = "Self Employed",rot=0)
print(df_x['Self_Employed'].value_counts())


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_x['Credit_History'] = le.fit_transform(df_x['Credit_History'].astype(str))


# In[30]:


plt.style.use('ggplot')
df_x['Credit_History'].value_counts().plot.bar(title='Credit History',rot=0)
print(df_x['Credit_History'].value_counts())


# # Visualizations

# In[31]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


import seaborn as sns
sns.set_style('dark')



# In[33]:


df_x.plot(figsize=(18, 8))

plt.show()


# In[34]:


plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicatoin Income vs Loan Amount ")

plt.scatter(df_x['ApplicantIncome'] , df_x['LoanAmount'])
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()


# In[35]:


sns.histplot(x='Gender',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# In[36]:


sns.histplot(x='Married',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# In[37]:


sns.histplot(x='Dependents',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# In[38]:


sns.histplot(x='Education',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# In[39]:


sns.histplot(x='Self_Employed',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# In[40]:


sns.histplot(x='Credit_History',palette='dark',data=df_x,hue=df_x['Loan_Status'])


# ### Encoding train data

# In[41]:


#Encoding required columns data
from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area"]
le = LabelEncoder()
for col in cols:
    df_x[col] = le.fit_transform(df_x[col])


# In[42]:


#dropping features which has no use
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term",'Loan_ID', 'CoapplicantIncome', 'Dependents']
df_x = df_x.drop(columns=cols, axis=1)


# ### Encoding test data

# In[43]:


#Encoding required columns data
cols = ['Gender',"Married","Education",'Self_Employed',"Credit_History","Property_Area"]
le = LabelEncoder()
for col in cols:
    df_y[col] = le.fit_transform(df_y[col])


# In[44]:


#dropping features which has no use
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term",'Loan_ID', 'CoapplicantIncome', 'Dependents']
df_y = df_y.drop(columns=cols, axis=1)


# ### Assigning cols to variables

# In[45]:


cols=['Credit_History', 'Education', 'Gender','Self_Employed']
X_train=df_x[cols].values
y_train=df_x['Loan_Status'].values


# In[46]:


X_test=df_y[cols].values


# In[47]:


le.fit(y_train)
y_train=le.transform(y_train)


# ## RandomForest

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


model=RandomForestClassifier()


# In[50]:


model.fit(X_train,y_train)


# In[51]:


y_pred=model.predict(X_test)


# In[52]:


from sklearn.metrics import accuracy_score
score = model.score(X_train, y_train)
print('accuracy_score overall :', score)
print('accuracy_score percent :', round(score*100,2))


# ## checking another model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model1=KNeighborsClassifier(n_neighbors=3)


# In[ ]:


model1.fit(X_train,y_train)


# In[ ]:


y_pred1=model1.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
score = model1.score(X_train, y_train)
print('accuracy_score overall :', score)
print('accuracy_score percent :', round(score*100,2))


# ## checking another model

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model2=LogisticRegression()


# In[ ]:


model2.fit(X_train,y_train)


# In[ ]:


y_pred2=model2.predict(X_test)


# In[ ]:


#y_pred2 = le.inverse_transform(y_pred2)


# In[ ]:


# df_y['Loan_Status']=y_pred2
# outcome_var = 'Loan_Status'

# df_y.to_csv("Logistic_Prediction.csv",columns=['Loan_Status'])


# In[ ]:


from sklearn.metrics import accuracy_score
score = model2.score(X_train, y_train)
print('accuracy_score overall :', score)
print('accuracy_score percent :', round(score*100,2))


# In[ ]:



st.header("Check if you are eligible to get a loan")
var1 = st.radio("Select Credit History: ", ('0', '1'))

var2 = st.selectbox("Education: ",['Graduated', 'UnGraduated'])

var3 = st.radio("Select Gender: ", ('Male', 'Female'))

var4 = st.selectbox("SelfEmployed: ",['Yes', 'No'])

if st.button("Predict"):
    y_pred2




# In[ ]:




