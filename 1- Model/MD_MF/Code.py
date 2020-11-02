###############################################################################
# Train and test dataset uploading  

from pandas import read_csv
import numpy as np,pandas as pd
X_train = open("X_train.csv") 
X_train = pd.read_csv('X_train.csv')
X_train.head(5)

X_test = open("X_test.csv") 
X_test = pd.read_csv('X_test.csv')
X_test.head(5)

y_train = open("y_train.csv") 
y_train = pd.read_csv('y_train.csv')
y_train.head(5)

y_test = open("y_test.csv") 
y_test = pd.read_csv('y_test.csv')
y_test.head(5)

###############################################################################
# The final parameters used for our model were obtained as follows.

from sklearn.ensemble import RandomForestRegressor
RandomForestRegressorModel = RandomForestRegressor(bootstrap=True, criterion='mse', 
                      max_depth=105.76470990275939,
                      max_features=0.21785338690592665, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=3,
                      min_weight_fraction_leaf=0.0, n_estimators=993,
                      n_jobs=None, oob_score=False, random_state=666, verbose=0,
                      warm_start=False)
RandomForestRegressorModel.fit(X_train, y_train)

#==============================================================================
#Calculating Prediction
y_pred0 = RandomForestRegressorModel.predict(X_train)
print('Predicted Value for Random Forest Regressor -Train Set- is: ' , y_pred0[:])
# Convert from numpy arrays to pandas dataframes 
y_pred0 = pd.DataFrame(y_pred0)
y_pred0.to_csv('tR Pred0.csv', index = True, header = True)


y_pred = RandomForestRegressorModel.predict(X_test)
print('Predicted Value for Random Forest Regressor -Test Set- is: ' , y_pred[:])
# Convert from numpy arrays to pandas dataframes 
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('tR Pred.csv', index = True, header = True)
#------------------------------------------------------------------------------
# Error Calculating

from sklearn.metrics import mean_absolute_error 
MAEValue = mean_absolute_error(y_train, y_pred0, multioutput='uniform_average') 
print('Mean Absolute Error Value for Training Set is: ', MAEValue)
MAEValuetest = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Absolute Error Value for Test Set is: ', MAEValuetest)
