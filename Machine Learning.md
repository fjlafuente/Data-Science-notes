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

#RMSE: It penalizes more high values of error:

from sklearn.metrics import mean_squared_error

#The function does not include the square, we have to add it manually:

np.sqrt(mean_squared_error(reg.predict(X_test), y_test))

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
````
## Cross Validation
![grid_search_cross_validation](https://user-images.githubusercontent.com/116038816/224561052-6ad7bc0f-a8d3-451e-831e-07d990811464.png)

"Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting".
A good way to test the data several times is calles Cross Validation. The data is split into k pieces, k-1 of them are traindata, 1 of them is test data. K iterations are made to the data and the average of the differences are calculated afterwards.

````python
#Library

from sklearn.model_selection import cross_val_score

#The function would be as follows, with cv as main parameter: is the number of parts in which the data is split.
#We have to include first the model, then the variables and then the parameters:
cross_val_score(reg, X, y, cv = k, scoring = 'neg_mean_squared_error')

#By default, the score computed at each CV iteration is the score method of the estimator. It is possible to change this by using the scoring parameter.
#KFold can aslo be interesting
````
## GridSearch

It is a common way to test several parameters at the same time and see which selection fits the better the model. For example, hoe many kneighbors are necessary.
We would include a dictionary of parameters:

````python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

reg_test = GridSearchCV(KneighborsRegressor(), param_grid = {'n_neighbors': np.arange(3,50)})

#FIt all the combinations of parameters possible

reg_test.fit(X,y)

#Best estimator and best parameters

results = pd.DataFrame(reg_test.cv_results) # Transform into df a Dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
results.to_csv('results.csv', index = False) #we can save the results into a file.
reg_test.best_score
reg_test.best_estimator
reg_test.best_params
````
## Decission Tree
A decision tree is a structure that includes a root node, branches and leaf nodes. Each internal node denotes a test on an attribute, each branch denotes the outcome of a test and each leaf node holds a class label. Topmost node in the tree is the root node.
It does not support empty values.
It builds homogeneus partitions of data based on simple decision rules inferred from the data features.
It splits data in 2 partitions and calculate the purity/homogeneity gain and repeat the process k times.
Homogeneity is calculated with the variance -regression- or entropy -classification-.






