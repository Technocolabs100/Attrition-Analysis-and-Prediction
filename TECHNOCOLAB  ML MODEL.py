#!/usr/bin/env python
# coding: utf-8

# # Import Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Ashmi\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[2]:


df.head()


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.describe()


# In[ ]:





# In[6]:


columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
ds = df.drop(columns=columns_to_drop)


# In[7]:


ds.shape


# In[8]:


import missingno as msno
missing_values = ds.isna().sum()
print(missing_values)
msno.matrix(ds)
plt.show()


# # Exploratory Data Analysis

# In[9]:


unique_values = {col: ds[col].unique() for col in ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                                                  'JobLevel','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','StockOptionLevel',
                                                  'WorkLifeBalance']}
print(unique_values)


# In[10]:


# Convert columns to categorical

ds['Education'] = ds['Education'].astype('category')
ds['EnvironmentSatisfaction'] = ds['EnvironmentSatisfaction'].astype('category')
ds['JobInvolvement'] = ds['JobInvolvement'].astype('category')
ds['JobLevel'] = ds['JobLevel'].astype('category')
ds['JobSatisfaction'] = ds['JobSatisfaction'].astype('category')
ds['PerformanceRating'] = ds['PerformanceRating'].astype('category')
ds['RelationshipSatisfaction'] = ds['RelationshipSatisfaction'].astype('category')
ds['StockOptionLevel'] = ds['StockOptionLevel'].astype('category')
ds['WorkLifeBalance'] = ds['WorkLifeBalance'].astype('category')


# In[11]:


# Select numerical columns
numerical_vars = ds.select_dtypes(include=['number'])

# Display the numerical columns
print(numerical_vars)


# In[12]:


# Select categorical columns (both object type and category type)
categorical_vars = ds.select_dtypes(include=['object', 'category'])

# Display the categorical columns
print(categorical_vars)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'ds' is your DataFrame

# Select numerical variables for bivariate analysis
numerical_vars_bivariate = ["Age", "DailyRate", "DistanceFromHome",
                            "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
                            "PercentSalaryHike", "TotalWorkingYears",
                            "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                            "YearsWithCurrManager"]

# Calculate the number of rows and columns for the plot layout
num_rows_num = -(-len(numerical_vars_bivariate) // 4)  # Equivalent to ceiling division
num_cols_num = min(len(numerical_vars_bivariate), 4)

# Set the figure size
plt.figure(figsize=(12, 5 * num_rows_num))

# Create box plots for each numerical variable
for i, var in enumerate(numerical_vars_bivariate):
    plt.subplot(num_rows_num, num_cols_num, i + 1)
    sns.boxplot(x='Attrition', y=var, data=ds, palette='Set3')
    plt.title(f'Attrition vs. {var}')
    plt.xlabel('Attrition')
    plt.ylabel(var)
    plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'ds' is your DataFrame

# Select categorical variables for bivariate analysis
categorical_vars_bivariate = ["BusinessTravel", "Department", "Education", "EducationField",
                              "EnvironmentSatisfaction", "Gender", "JobInvolvement",
                              "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
                              "OverTime", "PerformanceRating", "RelationshipSatisfaction",
                              "StockOptionLevel", "WorkLifeBalance"]

# Calculate the number of rows and columns for the plot layout
num_rows_cat = -(-len(categorical_vars_bivariate) // 2)  # Equivalent to ceiling division
num_cols_cat = 2

# Set the figure size
plt.figure(figsize=(12, 5 * num_rows_cat))

# Create stacked bar plots for each categorical variable
for i, var in enumerate(categorical_vars_bivariate):
    plt.subplot(num_rows_cat, num_cols_cat, i + 1)
    sns.countplot(data=ds, x=var, hue='Attrition', palette='Set3')
    plt.title(f'Attrition by {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Attrition', loc='upper right', labels=['Not Attrited', 'Attrited'])

# Adjust layout
plt.tight_layout()
plt.show()


# In[15]:


ds['Attrition'] = ds['Attrition'].astype('category').cat.codes
ds['Education'] = ds['Education'].astype('category').cat.codes
ds['EnvironmentSatisfaction'] = ds['EnvironmentSatisfaction'].astype('category').cat.codes
ds['PerformanceRating'] = ds['PerformanceRating'].astype('category').cat.codes
ds['RelationshipSatisfaction'] = ds['RelationshipSatisfaction'].astype('category').cat.codes
ds['JobSatisfaction'] = ds['JobSatisfaction'].astype('category').cat.codes
ds['JobInvolvement'] = ds['JobInvolvement'].astype('category').cat.codes
ds['JobLevel'] = ds['JobLevel'].astype('category').cat.codes
ds['Gender'] = ds['Gender'].astype('category').cat.codes
ds['BusinessTravel'] = ds['BusinessTravel'].astype('category').cat.codes
ds['Department'] = ds['Department'].astype('category').cat.codes
ds['Over18'] = ds['Over18'].astype('category').cat.codes
ds['OverTime'] = ds['OverTime'].astype('category').cat.codes
ds['StockOptionLevel'] = ds['StockOptionLevel'].astype('category').cat.codes
ds['WorkLifeBalance'] = ds['WorkLifeBalance'].astype('category').cat.codes


# In[16]:


ds


# In[18]:


from sklearn.preprocessing import OneHotEncoder

# Select the categorical columns
ds1 = ds[['EducationField', 'MaritalStatus', 'JobRole']]

# Select the remaining columns
ds2 = ds.drop(columns=['EducationField', 'MaritalStatus', 'JobRole'])

# Perform one-hot encoding
encoder = OneHotEncoder(sparse=False)  # Set sparse=False to return a dense array
onehot = encoder.fit_transform(ds1)

# Convert the array back to a DataFrame
onehot_df = pd.DataFrame(onehot, columns=encoder.get_feature_names_out(ds1.columns)).astype(int)
# Combine one-hot encoded columns with the rest of the data
final_df = pd.concat([ds2, onehot_df], axis=1)

# Display the first few rows of the final DataFrame
print(final_df.head())


# # Building ML Models

# In[49]:


corr = final_df.corr()


# In[55]:


# correlation matrix
plt.figure(figsize=(25, 16))
ax = sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[19]:


from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron,SGDClassifier,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV, learning_curve, cross_val_score


# In[20]:


#Separating Feature and Target matrices
X = final_df.drop(['Attrition'], axis=1)
y=final_df['Attrition']


# In[21]:


X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)


# In[22]:


print(f"Train data shape: {X_train.shape}, Test Data Shape {X_test.shape}")


# In[23]:


# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train,y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo,X_train,y_train,cv=cv,n_jobs = -1)
    
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# In[24]:


# Logistic Regression
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), X_train,y_train, 10)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)


# In[25]:


# SVC
train_pred_svc, acc_svc, acc_cv_svc = fit_ml_algo(SVC(),X_train,y_train,10)
print("Accuracy: %s" % acc_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_svc)


# In[26]:


# Gaussian Naive Bayes

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(),X_train,y_train,10)

print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)


# In[27]:


# Decision Tree

train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(),X_train, y_train,10)

print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)


# In[28]:


# Random Forest

train_pred_dt, acc_rf, acc_cv_rf = fit_ml_algo(RandomForestClassifier(n_estimators=100),X_train, y_train,10)

print("Accuracy: %s" % acc_rf)
print("Accuracy CV 10-Fold: %s" % acc_cv_rf)


# In[29]:


import xgboost as xgb
train_pred_dt, acc_xg, acc_cv_xg = fit_ml_algo(xgb.XGBClassifier(random_state=42),X_train, y_train,10)
print("Accuracy: %s" % acc_xg)
print("Accuracy CV 10-Fold: %s" % acc_cv_xg)


# In[30]:


svcreg = SVC()
svcreg.fit(X_train, y_train)


# In[31]:


#PRScores for Logistic Regression
print(classification_report(y_test, svcreg.predict(X_test)))


# In[32]:


loreg = LogisticRegression()
loreg.fit(X_train, y_train)


# In[33]:


#PRScores for Logistic Regression
print(classification_report(y_test, loreg.predict(X_test)))


# In[34]:


gnbreg = GaussianNB()
gnbreg.fit(X_train, y_train)


# In[35]:


#PRScores for Logistic Regression
print(classification_report(y_test, gnbreg.predict(X_test)))


# In[36]:


dtcreg = DecisionTreeClassifier()
dtcreg.fit(X_train, y_train)


# In[37]:


#PRScores for Logistic Regression
print(classification_report(y_test, dtcreg.predict(X_test)))


# In[38]:


rfcreg = RandomForestClassifier(n_estimators=100)
rfcreg.fit(X_train, y_train)


# In[39]:


#PRScores for Logistic Regression
print(classification_report(y_test, rfcreg.predict(X_test)))


# In[40]:


xgbreg = xgb.XGBClassifier(random_state=42)
xgbreg.fit(X_train, y_train)


# In[41]:


#PRScores for Logistic Regression
print(classification_report(y_test, xgbreg.predict(X_test)))


# In[42]:


model = LogisticRegression().fit(X_train, y_train)


# In[43]:


predictions = model.predict(X_test)


# In[44]:


pred_df = pd.DataFrame(index=X_test.index)


# In[45]:


pred_df['Attrition'] = predictions
pred_df


# In[46]:


final_df.insert(1, "Predictions", model.predict(X))
final_df

