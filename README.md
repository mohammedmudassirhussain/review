import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/content/finaldatatrain.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
df['LoanAmount'] = df['LoanAmount']
df['Credit_History'] = df['Credit_History']
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.shape
plt.figure(figsize = (100, 50))
sns.set(font_scale = 5)
plt.subplot(331)
sns.countplot(df['Gender'],hue=df['Loan_Status'])

plt.subplot(332)
sns.countplot(df['Married'],hue=df['Loan_Status'])

plt.subplot(333)
sns.countplot(df['Education'],hue=df['Loan_Status'])

plt.subplot(334)
sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])

plt.subplot(335)
sns.countplot(df['Property_Area'])
df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].value_counts()
df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()
df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()
df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()
df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()

df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()
df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df['Property_Area'].value_counts()
df['LoanAmount'].value_counts()
df['Loan_Amount_Term'].value_counts
df['Credit_History'].value_counts()
df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
x = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))
print("y_predicted",lr_prediction)
print("y_test",y_test)
