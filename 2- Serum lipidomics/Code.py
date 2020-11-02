###############################################################################
# data uploading  

from pandas import read_csv
import numpy as np,pandas as pd
MD = open("MD.csv") 
MD = pd.read_csv('MD.csv')
MD.head(5)

MF = open("MF.csv") 
MF = pd.read_csv('MF.csv')
MF.head(5)

MD_MF = open("MD_MF.csv") 
MD_MF = pd.read_csv('MD_MF.csv')
MD_MF.head(5)

tR_Exp = open("tR_Exp.csv") 
tR_Exp = pd.read_csv('tR_Exp.csv')
tR_Exp.head(5)


###############################################################################
# applying (external validation set)

import sklearn.externals.joblib as jb
ModelMD = jb.load('RandomForestRegressorModelMD.sav')
ModelMF = jb.load('RandomForestRegressorModelMF.sav')
ModelMD_MF = jb.load('RandomForestRegressorModelMD_MF.sav')

#==============================================================================
#Calculating Prediction
y_predMD = ModelMD.predict(MD)
print('Predicted Value for Random Forest Regressor -MD- is: ' , y_predMD[:])
# Convert from numpy arrays to pandas dataframes 
y_predMD = pd.DataFrame(y_predMD)
y_predMD.to_csv('y_predMD.csv', index = True, header = True)

y_predMF = ModelMF.predict(MF)
print('Predicted Value for Random Forest Regressor -MF- is: ' , y_predMF[:])
# Convert from numpy arrays to pandas dataframes 
y_predMF = pd.DataFrame(y_predMF)
y_predMF.to_csv('y_predMF.csv', index = True, header = True)

y_predMD_MF = ModelMD_MF.predict(MD_MF)
print('Predicted Value for Random Forest Regressor -MD_MF- is: ' , y_predMD_MF[:])
# Convert from numpy arrays to pandas dataframes 
y_predMD_MF = pd.DataFrame(y_predMD_MF)
y_predMD_MF.to_csv('y_predMD_MF.csv', index = True, header = True)
#------------------------------------------------------------------------------
# Error Calculating

from sklearn.metrics import mean_absolute_error 
MAEValueMD = mean_absolute_error(tR_Exp, y_predMD, multioutput='uniform_average') 
print('Mean Absolute Error Value for MD is: ', MAEValueMD)
MAEValueMF = mean_absolute_error(tR_Exp, y_predMF, multioutput='uniform_average') 
print('Mean Absolute Error Value for MF is: ', MAEValueMF)
MAEValueMD_MF = mean_absolute_error(tR_Exp, y_predMD_MF, multioutput='uniform_average') 
print('Mean Absolute Error Value for MD_MF is: ', MAEValueMD_MF)
