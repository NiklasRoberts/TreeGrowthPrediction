# importing relevant packages
print("Importing relevant packages.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# Get tree measurements for year 82, 85, and 90
print("Importing and cleaning tree measurements.")
treeMeasurements = pd.read_csv("C55_TreeMeasurements.csv")


# getting trees from 1982 and 1985 measurements
desiredCols1 = ["PLOT", "TREE", "RECRUIT", "SPECIES", "D82", "D85"]
customMeasurements1 = treeMeasurements[desiredCols1]

# getting trees from 1985 to 1990
desiredCols2 = ["PLOT", "TREE", "RECRUIT", "SPECIES", "D85", "D90"]
customMeasurements2 = treeMeasurements[desiredCols2]

# Dropping rows without values or "na"
# getting trees that only have measurements at both ends of the period
clean82_85 = customMeasurements1.dropna()
clean85_90 = customMeasurements2.dropna()

# getting foliar nutrients from csv file
print("Importing Foliar Nutrients, appending to tree data.")
foliarNutrients = pd.read_csv("C55_FoliarNutrients.csv")
foliarNutrients = foliarNutrients.drop("REPLICATE", axis=1)

# Get foliar nutrient levels 1982
foliarNutrients82 = foliarNutrients.loc[foliarNutrients['YEAR'] == 1982]
# Get foliar nutrient levels 1985
foliarNutrients85 = foliarNutrients.loc[foliarNutrients['YEAR'] == 1985]

# Add foliar nutrients to tree data 
allTreeData82_85 = pd.merge(clean82_85, foliarNutrients82, on="PLOT", how="outer")
allTreeData85_90 = pd.merge(clean85_90, foliarNutrients85, on="PLOT", how="outer")

# Getting percent growth for each tree per year
# perc_growth = (D85-D82)/D82
print("Calculating growth percent per year.")
allTreeData82_85["Growth"] = (allTreeData82_85["D85"]-allTreeData82_85["D82"])/allTreeData82_85["D82"]
# growth_per_yr = prec_growth/85-82
allTreeData82_85["Growth/yr"] = allTreeData82_85["Growth"]/3

#getting percent growth for 1985-1990
# perc_growth = (D90-D85)/D825
allTreeData85_90["Growth"] = (allTreeData85_90["D90"]-allTreeData85_90["D85"])/allTreeData85_90["D85"]
# growth_per_yr = prec_growth/85-82
allTreeData85_90["Growth/yr"] = allTreeData85_90["Growth"]/5

# Extract relevant columns: 
relColumns = ["SPECIES", "TOTN", "TOTP", "TOTK", "TOTS", "TOTCa", "Growth/yr"]
modelingData82 = allTreeData82_85[relColumns]
modelingData85 = allTreeData85_90[relColumns]

# combine dataframes to make one dataframe with all data needed
modelingData = pd.concat([modelingData82, modelingData85], ignore_index=True)
modelingData['id'] = modelingData.index


# Getting Columns for neural network regression
# input variables: SPECIES, TOTN, TOTP, TOTK, TOTS, TOTCa, and TOTMg (not used)
# label: predicting growth per year
print("Extracting columns for regression.")
regressColumns = ['SPECIES', 'TOTN', 'TOTP', 'TOTK', 'TOTS', 'TOTCa', 'Growth/yr']
regressData = modelingData[regressColumns]

# for turning 'species' column into numerical data
speciesLabEncoder = LabelEncoder()
# Transforms species into numerical data
speciesNumList = speciesLabEncoder.fit_transform(regressData['SPECIES'])
# code for reverting: species = le.inverse_transform(label)
# changing the species column to numerical values
regressData['SPECIES'] = speciesNumList

# Splitting into input and output data
regressX = regressData.iloc[:,:6].values
regressY = regressData.iloc[:,6:7].values


# Normalizing and scaling input/output variables
regressY = np.reshape(regressY, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print (scaler_x.fit(regressX))
xscale = scaler_x.transform(regressX)
print (scaler_y.fit(regressY))
yscale = scaler_y.transform(regressY)


# Splitting data into training and test data
regressX_train, regressX_test, regressY_train, regressY_test = train_test_split(xscale, yscale, test_size = 0.2)

#Building and Compiling Neural Network for Regression
# 6/8/1 structure, relu activations, adam optimizer, loss function is mean absolute error
regressModel = Sequential()
regressModel.add(Dense(8, input_dim = 6, activation='relu'))
regressModel.add(Dense(1, activation ='relu'))
#loss function is based of mean absolute error
loss_function = keras.losses.MeanAbsoluteError()
regressModel.compile(loss=loss_function, optimizer='adam', metrics=['mse','mae','mape'])

# training model
regressModel.fit(regressX_train, regressY_train, epochs=100, batch_size=64)

# making predictions
prediction = regressModel.predict(regressX_test)

# inverse transform of data
pred = scaler_y.inverse_transform(prediction)
true = scaler_y.inverse_transform(regressY_test)

# Metrics
print("mean Squared Error: ", mean_squared_error(true, pred))
print("Mean Absolute Error: ", mean_absolute_error(true, pred))
print("Max Error: ", max_error(true, pred))
absErrors = abs(true - pred)

# making a histogram
plt.figure()
plt.hist(absErrors, bins = 20)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Absolute Error Distribution")
plt.savefig('6_Relu_8_Relu_1_Full_Histogram.png')

#plt.show()


#plt.figure()
#plt.hist(absErrors, bins = 10)
#plt.xlim(0,0.5)
#plt.xlabel("Absolute Error")
#plt.ylabel("Frequency")
#plt.title("Absolute Error Distribution 0 to 0.5")
#plt.savefig('6_Relu_8_Relu_1_FirstHalf_Histogram.png')

#plt.figure()
#plt.hist(absErrors, bins = 11)
#plt.xlim(0.25,4)
#plt.ylim(0,100)
#plt.xlabel("Absolute Error")
#plt.ylabel("Frequency")
#plt.title("Absolute Error Distribution 0.25 to 4")
#plt.savefig('6_Relu_8_Relu_1_SecondHalf_Histogram.png')

# gathering values to help determine which model is best
errorDistribution = []
errorDistribution.append(len([i for i in absErrors if  i <= 0.001]))
errorDistribution.append(len([i for i in absErrors if  i > 0.001 and i <= 0.005]))
errorDistribution.append(len([i for i in absErrors if  i > 0.005 and i <= 0.01]))
errorDistribution.append(len([i for i in absErrors if  i > 0.01 and i <= 0.025]))
errorDistribution.append(len([i for i in absErrors if  i > 0.025 and i <= 0.05]))
errorDistribution.append(len([i for i in absErrors if  i > 0.05 and i <= 0.1]))
errorDistribution.append(len([i for i in absErrors if  i > 0.1 and i <= 0.25]))
errorDistribution.append(len([i for i in absErrors if  i > 0.25 and i <= 0.5]))
errorDistribution.append(len([i for i in absErrors if  i > 0.5]))

# printing out message to help with analysis
total = len(absErrors)
print("Difference of percent growth between prediction and true")
print("Total number of entries: ", total)
print("<=0.001 diff :    ", errorDistribution[0], "     Percent: ", errorDistribution[0]/total * 100)
print("0.001 to 0.005 :  ", errorDistribution[1], "     Percent: ", errorDistribution[1]/total * 100)
print("0.005 to 0.1 :    ", errorDistribution[2], "     Percent: ", errorDistribution[2]/total * 100)
print("0.1 to 0.025 :    ", errorDistribution[3], "     Percent: ", errorDistribution[3]/total * 100)
print("0.25 to 0.05 :    ", errorDistribution[4], "     Percent: ", errorDistribution[4]/total * 100)
print("0.05 to 0.10 :    ", errorDistribution[5], "     Percent: ", errorDistribution[5]/total * 100)
print("0.10 to 0.25 :    ", errorDistribution[6], "     Percent: ", errorDistribution[6]/total * 100)
print("0.25 to 0.50 :    ", errorDistribution[7], "     Percent: ", errorDistribution[7]/total * 100)
print("> 0.50 :          ", errorDistribution[8], "     Percent: ", errorDistribution[8]/total * 100)