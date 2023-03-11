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

#Our training set error will always be an optimistic estimate for our test set error. Therefore we have to split our data into train data and test data:

from sklearn.model_selection import train_test_split

#We have to define two groups with variable X and y in each group:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 4)

#train_test_split params:
  #test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If None, the value is set to the  complement of the train size. If train_size is also None, it will be set to 0.25.
  #random_state: Controls the shuffling applied to the data before applying the split

#Librery sklearn linear regression

from sklearn.linear_model import LinearRegression

#Create an instance for the model

reg = LinearRegression()

#Fit the regressor. We train the model based on actual data (train data). It is always done with train data

reg.fit (X_train,y_train) #X always upper, y always lower

#Evaluation of the model: MAE. Predict method allows us to make predictions using our model.

from sklearn.metrics import mean_absolute_error

mean_absolute_error(reg.predict(X_test),y_test)

#for MAPE (same as MAE but in percentage):

np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)

````

## K Nearest Neighbors 

Alorithm based supervised marchine learning tool. It requires that the dataset has a distance.
It can classify the data based on the k-neighbors of a given point: it analyses which are the characteristics of the closest points and uses the info to predict and classify the given point.
'K' is obviously going to be the key factor in the analysys and it is difficullt to know hoe many neighbors are the correct: too less neighbors is very unprecise - analysys made comparing less data- and so it does many neighbors - it is difficult to classify the items-.

````python

#Load the librery from sklearn

from sklearn.neighbors import KNeighborsRegressor

#Create an instance for the model

regk = KNeighborsRegressor()

#Fit the data. NOT train phase in this method. It takes all the data for the analysis.

regk.fit(X,y) 





