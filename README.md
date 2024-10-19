# IBM-PROJECT

## AIM

To Predict Employee Attrition in an Organization.

## Problem Statement and Dataset

The goal is to predict employee attrition in an organization. Employee attrition refers to the departure of employees from a company, which can lead to increased costs for recruitment, training, and reduced productivity. By predicting which employees are likely to leave, HR teams can take proactive measures to retain talent, improve job satisfaction, and optimize workforce management.

-Contains features related to employee demographics, job roles, financial compensation, satisfaction, and performance.

-Target Variable: Attrition (Yes/No)

## DESIGN STEPS

### step 1:

Data Understanding is used to review dataset structure and identify data types.

### step 2:

Data Preprocessing Handle missing values, convert categorical variables, and remove duplicates.

### step 3:

Exploratory Data Analysis (EDA) is to visualize demographic distributions and analyze attrition rates.

### step 4:

Feature Engineering Create interaction terms and categorize variables.

### step 5:

Model Selection is used to Split data into training/testing sets and choose machine learning models.

### step 6:

Model Training is used to Fit models, tune hyperparameters, and track performance metrics.

### step 7:

Model Evaluation is used to Analyze model performance using confusion matrices and metrics.

### step 8:

Interpretation and Insights used to Identify key features affecting attrition and provide recommendations.

### step 9:

Reporting to Summarize findings in a report and create visualizations.

### step 10:

Implementation and Monitoring for Collaborate on retention strategies and monitor ongoing attrition rates.

## PROGRAM


````
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('employee_attrition.csv')
df.head()

df.describe()

df.info()

att_cnt = pd.DataFrame(df['Attrition'].value_counts())
att_cnt

df.drop(['EmployeeCount','EmployeeNumber'],axis=1,inplace=True)
df.head()

att_dummies = pd.get_dummies(df['Attrition'], dtype='int')
att_dummies.head()

df = pd.concat([df,att_dummies],axis=1)
df.head()

df = df.drop(['Attrition','No'],axis=1)
df.head()

sns.barplot(x=df['Gender'],y=df['Yes'])

sns.barplot(x=df['Department'],y=df['Yes'])

sns.heatmap(df.corr(numeric_only=True))

df = df.drop(['Age','JobLevel'],axis=1)

from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    if df[column].dtype==np.number:
        continue
    else:
        df[column]=LabelEncoder().fit_transform(df[column])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)

X = df.drop(['Yes'],axis=1)
y = df['Yes']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

rf.fit(X_train,y_train)

rf.score(X_train,y_train)

pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


````

## OUTPUT

### bar plot for gender for attirition

![image](https://github.com/user-attachments/assets/ba650a94-7140-4436-aa1e-16c0ec5105e1)

### bar plot for department for employee attirition

![image](https://github.com/user-attachments/assets/2cfef56a-be13-469b-871b-eda8c4ae2e59)

### graph for employee

![image](https://github.com/user-attachments/assets/970607c5-be17-4e8a-8ad4-5bcd8c58cb67)

## RESULT

The analysis of the employee attrition dataset will yield insights into demographic patterns, key factors influencing attrition (like job satisfaction and salary), and visualizations of these relationships. Model performance metrics will assess predictive accuracy, while actionable recommendations will target high-risk groups to improve retention. A comprehensive report will summarize findings and outline an implementation plan for monitoring retention strategies.


