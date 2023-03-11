# MACHINE LEARNING

Modules explained in this guide mainly related to sklearn.

![sklearn](https://user-images.githubusercontent.com/116038816/224510655-264a75d9-ce05-422e-82be-4cba53d60cf2.png)

## SUPERVISED LEARNING

### Linear Regression

Linear equation with one or more variables that is used for predicting the value of a dependant variable.

````python
#First we have to define our data

df = pd.read_csv('test.csv')
X = df['variable 1']
y = df['variable 2 target']

#Our training set error will always be an optimistic estimate for our test set error.

#Librery sklearn linear regression

from sklearn.linear_model import LinearRegression

#Create an instance for the model

reg = LinearRegression()

#Fit the regressor. We train the model based on actual data (train data)

reg.fit (X,y) #X always upper, y always lower








