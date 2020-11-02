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
# Random Forest + CV 

## installing bayesian-optimization (https://pypi.org/project/bayesian-optimization/)
#  pip install bayesian-optimization
'''

# Parameter tuning is an essential step for accurate model realization. Many modern ML algorithms have a large number of parameters. 
# To efficiently use RF, we need to select suitable parameter values. Bayesian Optimization is a suitable technique often used for parameter tuning, therefore, we used it here for getting the best parameter.

def trregressor(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    def RandomForestRegressorModel_cv(n_estimators, min_samples_split, max_features, max_depth):
        RandomForestRegressorModel = RandomForestRegressor(n_estimators=int(n_estimators),
                                    min_samples_split=int(min_samples_split),
                                    max_features=min(max_features, 0.999),  
                                    max_depth=int(max_depth),
                                    random_state=0,  
                                    n_jobs=-1)
        scores = cross_val_score(RandomForestRegressorModel, X_train, y_train, cv=10)
        score = scores.mean()
        return score
    from bayes_opt import BayesianOptimization
    RandomForestRegressorModel_bo = BayesianOptimization(
        RandomForestRegressorModel_cv,
        {
            'n_estimators': (10, 1000),
            'min_samples_split': (2, 100),
            'max_features': (0.1, 0.999),
            'max_depth': (5, 300)
        }
    )
    RandomForestRegressorModel_bo.maximize()

    best_max_depth = RandomForestRegressorModel_bo.res[0]['params']['max_depth']
    best_max_features = RandomForestRegressorModel_bo.res[0]['params']['max_features']
    best_min_samples_split = RandomForestRegressorModel_bo.res[0]['params']['min_samples_split']
    best_n_estimators = RandomForestRegressorModel_bo.res[0]['params']['n_estimators']
    highest_score = RandomForestRegressorModel_bo.res[0]['target']

    for i in RandomForestRegressorModel_bo.res:
        if i['target'] > highest_score:
            highest_score = i['target']
            best_max_depth = i['params']['max_depth']
            best_max_features = i['params']['max_features']
            best_min_samples_split = i['params']['min_samples_split']
            best_n_estimators = i['params']['n_estimators']

    RandomForestRegressorModel = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features,
                                min_samples_split=best_min_samples_split.astype(int),
                                n_estimators=best_n_estimators.astype(int), random_state=666)
    return RandomForestRegressorModel

RandomForestRegressorModel=trregressor(X_train, y_train)
RandomForestRegressorModel.fit(X_train, y_train)
'''
#==============================================================================
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
