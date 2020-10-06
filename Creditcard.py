import numpy as np, pandas as pd, sklearn, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #Transforms features by scaling each feature to a given range.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, r2_score,fbeta_score

# http://archive.ics.uci.edu/ml/datasets/credit+approval

columns = ['Male', 'Age', 'Debt', 'Married', 'BankCustomer','EducationLevel',
           'Enthicity', 'YearsEmployed','PriorDefault','Employed','CreditScore',
           'DriversLicense','Citizen','Zipcode','Income','Approved']
df = pd.read_csv('crx.data',header=None,names=columns, dtype=object)
df.to_csv('Credit.csv')

#printing heads values
print('Dataframe head:')
print(df.head())
print('Dataframe Description:')
print(df.describe())

# printing dataframe info
print('Dataframe INFO:')
print(df.info())# it shows columns along with values stored in that column

#################################Preprocessinging#################################

# checking for null values
print('Null Values:')
print(df.isnull().sum())

# Missing value
print('Values with "?":')
print(df.isin(['?']).sum())

# Replace "?" with NaN
print('Interchanging "?" with np.NaN')
df.replace('?', np.NaN, inplace = True)

# Rechecking
print('Values with "?"')
print(df.isin(['?']).sum())

# printing dataframe info
print('Dataframe INFO:')
print(df.info())

list_=['Age','Debt','YearsEmployed','Zipcode','Income']
for x in list_:
    df[x] = df[x].astype(float)

# printing dataframe info
print('Dataframe INFO:')
print(df.info())

# checking for null values
print('Null Values:')
print(df.isnull().sum())

# replacing numerical NaN with mean column value

df.fillna(df.mean(), inplace=True)

list_ = ['CreditScore','Zipcode','Income']
for x in list_:
    df[x] = df[x].astype(int)
    
# printing dataframe info
print('Dataframe INFO:')
print(df.info())

# checking for null values
print('Null Values:')
print(df.isnull().sum())

##Generating radom null values in data [3% missing value]
##df = df.stack().sample(frac=0.97).unstack().reindex(index=df.index, columns=df.columns)
##changing data types
##df.astype({'Age':float,'Debt':float,'YearsEmployed':float,'CreditScore':int,'Zipcode':int,'Income':int,})

print('Working on object dtype columns of DataFrame')
# Filling Null in object type Columns with most frequent value occuring value
print('Filling Null in object type Columns with most frequent value occuring value')
for colmn in df:
    if df[colmn].dtypes == 'object':
        df[colmn] = df[colmn].fillna(df[colmn].mode().iloc[0])

print('Null after operation on object dtype columns')
print(df.isnull().sum())

print('DataFrame Heads:')
print(df.head())

# Using label encoder to convert object type classification into numeric types
print('Using label encoder to convert object type classification into numeric types')
for colmn in df:
    if df[colmn].dtypes=='object':
        df[colmn]=LabelEncoder().fit_transform(df[colmn])

print('DataFrame Head after label encoder operation')
print(df.head())
        
#################################DataVisulations#################################

plt.figure()
sns.distplot(df['Age'], color = 'r')
plt.figure()
sns.distplot(df['Debt'], color = 'b')
plt.figure()
sns.distplot(df['YearsEmployed'], color = 'g')
plt.figure()
sns.distplot(df['CreditScore'], color = 'pink')
plt.figure()
sns.distplot(df['Income'], color = 'yellow')
plt.show()

#correlation
corln = df.corr()
sns.heatmap(corln, square=True)
plt.show()

columns_list = ['Age', 'Income', 'CreditScore', 'Debt', 'YearsEmployed']
sns.pairplot(df[columns_list],size=2)#height
plt.show()

sns.countplot(x = 'Approved',data = df)
plt.show()

#################################MakeingModel#################################

# Removing Less usefull or low value coloumns
df = df.drop(['DriversLicense', 'Zipcode'], axis=1)

# Dataframe to matrix formate
dtaMat = df.values

# Asinging Approved to labels
X,y = dtaMat[:,:13], dtaMat[:,13]

# Spliting Dataset
xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=0)

# Defining Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
sXtrain = scaler.fit_transform(xtrain)
sXtest = scaler.transform(xtest)

# Declearing Classifier
MdlBase = RandomForestClassifier(n_estimators=500)
# Inputing Data into Model
Model = MdlBase.fit(sXtrain, ytrain)

yPred = Model.predict(sXtest)
print('Random Forest classifier has accuracy of: ',Model.score(sXtest, ytest))
print('fbeta_score: ',fbeta_score(ytest, yPred, beta = 0.5))
print('r2_score: ', r2_score(ytest, yPred))
print('Confusion Matrix: \n',confusion_matrix(ytest, yPred))

#################################PlotModelParameter#################################

# Plot Confusion matrix
ax = plt.subplot()
sns.heatmap(confusion_matrix(ytest, yPred), annot=True, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
plt.show()

# Plot features importances
df_ = df.drop(['Approved'], axis=1)
features= df_.columns
importances = Model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
