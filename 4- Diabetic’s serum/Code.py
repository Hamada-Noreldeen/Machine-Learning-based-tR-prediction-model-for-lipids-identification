###############################################################################
# data uploading  

from pandas import read_csv
import numpy as np,pandas as pd
MD = open("MD.csv") 
MD = pd.read_csv('MD.csv')
MD.head(5)

tR_Exp = open("tR_Exp.csv") 
tR_Exp = pd.read_csv('tR_Exp.csv')
tR_Exp.head(5)

###############################################################################
# applying (transferring)

import sklearn.externals.joblib as jb
Model = jb.load('RandomForestRegressorModelMD.sav')

#==============================================================================
#Calculating Prediction
y_predMD = Model.predict(MD)
print('Predicted Value for Random Forest Regressor -MD- is: ' , y_predMD[:])
# Convert from numpy arrays to pandas dataframes 
y_predMD = pd.DataFrame(y_predMD)
y_predMD.to_csv('y_predMD.csv', index = True, header = True)
