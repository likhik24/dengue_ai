###### Authors: Likhitha , Sai Praveen, Sandhya
## Importing the required modules

from sklearn.ensemble import RandomForestRegressor
from ipykernel import kernelapp as app
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy
from sklearn import neighbors
from sklearn import linear_model
import csv
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
## Reading the Input Data
input_data = pd.read_csv("dengue_features_train.csv")
label_data = pd.read_csv("dengue_labels_train.csv")
## Reading the Test Data
test_data = pd.read_csv("dengue_features_test.csv")


##preprocessing test dataset as Attributes usually follow the most recent trends, so replacing the NA values with most recent values.
test_data.fillna(method='ffill',inplace=True)
##checking the count of null values  in test_data
# print(test_data.isnull().sum())


## Converting the Kelvin temperature to Celsius
def conversion(x):
    x = x -273.15
    return float(x)

## Merging the training label and training feature dataset
merged_train = pd.merge(input_data,label_data,on=('city','year','weekofyear'))
#print(merged_train.shape)

## preprocessing the training dataset
## Converting the Kelvin temperature to Celsius

def conversion(x):
    x = x -273.15
    return float(x)
conversion_columns = ["reanalysis_air_temp_k", "reanalysis_avg_temp_k","reanalysis_dew_point_temp_k","reanalysis_max_air_temp_k","reanalysis_min_air_temp_k"]
for column in conversion_columns:
    merged_train[column]=merged_train[column].apply(conversion)
    test_data[column]=test_data[column].apply(conversion)



lag_list = ["total_cases","Lag1","Lag2", "Lag3", "Lag4", "Lag5"]
for i in lag_list:
    test_data[i] = [0] * len(test_data)

## Partitioning the train, test dataset based on cities
sj_train = merged_train.loc[merged_train["city"] == "sj"]

iq_train = merged_train.loc[merged_train["city"] == "iq"]

## Converting the Week Start Date to a time series object
test_data['week_start_date']= pd.to_datetime(test_data.week_start_date)

sj_test = test_data.loc[test_data["city"] == "sj"]

iq_test = test_data.loc[test_data["city"] == "iq"]


## Attributes usually follow the most recent trends, so replacing the NA values with most recent values.
sj_train.fillna(method='ffill',inplace=True)
iq_train.fillna(method='ffill',inplace=True)
sj_train = sj_train.dropna()
iq_train = iq_train.dropna()
#print(sj_train_drop.shape)
#print(iq_train_drop.shape)

## Converting the Week Start Date to a time series object
sj_train['week_start_date'] = pd.to_datetime(sj_train.week_start_date)
sj_train['week_start_date'] = sj_train['week_start_date'].dt.strftime('%d-%m-%Y')
iq_train['week_start_date'] = pd.to_datetime(iq_train.week_start_date)
iq_train['week_start_date'] = iq_train['week_start_date'].dt.strftime('%d-%m-%Y')

#print(sj_train['week_start_date'])
#print(iq_train['week_start_date'])

input_features = ["precipitation_amt_mm","reanalysis_air_temp_k","reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k","reanalysis_max_air_temp_k","reanalysis_min_air_temp_k",
                "reanalysis_precip_amt_kg_per_m2","reanalysis_relative_humidity_percent",
                "reanalysis_sat_precip_amt_mm","reanalysis_specific_humidity_g_per_kg",
                "reanalysis_tdtr_k","station_avg_temp_c","station_diur_temp_rng_c",
                "station_max_temp_c","station_min_temp_c","station_precip_mm","ndvi_ne","ndvi_nw",
                  "ndvi_se","ndvi_sw","weekofyear", "city", "week_start_date","year"]
output_features = ["total_cases","city", "week_start_date","year"]

## Categorical Features are excluded while training
X_sj = sj_train.drop(output_features,axis=1).values
y_sj = sj_train.drop(input_features,axis=1).values
##Checking the important features using RandomForest Regressor
regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)
regr.fit(X_sj, y_sj.ravel())
sj_reg_col = sj_train.drop(output_features,axis=1).columns.values
#print(regr.feature_importances_)
#print(sj_reg_col)


## Removing the unwanted features/ features with high correlation
unwanted_features = ['city','reanalysis_tdtr_k','reanalysis_relative_humidity_percent',
                 'reanalysis_specific_humidity_g_per_kg','ndvi_ne','station_diur_temp_rng_c']
sj_train_processed = sj_train.drop(unwanted_features,axis=1)
iq_train_processed = iq_train.drop(unwanted_features,axis=1)
sj_test_data_processed = sj_test.drop(unwanted_features,axis=1)
iq_test_data_processed = iq_test.drop(unwanted_features,axis=1)

## Adding more lag features to make accurate predictions
sj_train_processed['Lag1']=sj_train_processed.total_cases.shift(1)
sj_train_processed['Lag2']=sj_train_processed.total_cases.shift(2)
sj_train_processed['Lag3']=sj_train_processed.total_cases.shift(3)
sj_train_processed['Lag4']=sj_train_processed.total_cases.shift(4)
sj_train_processed['Lag5']=sj_train_processed.total_cases.shift(5)
iq_train_processed['Lag1']=iq_train_processed.total_cases.shift(1)
iq_train_processed['Lag2']=iq_train_processed.total_cases.shift(2)
iq_train_processed['Lag3']=iq_train_processed.total_cases.shift(3)
iq_train_processed['Lag4']=iq_train_processed.total_cases.shift(4)
iq_train_processed['Lag5']=iq_train_processed.total_cases.shift(5)



input_features = ["precipitation_amt_mm","reanalysis_air_temp_k","reanalysis_avg_temp_k",
                  "reanalysis_dew_point_temp_k","reanalysis_max_air_temp_k","reanalysis_min_air_temp_k",
                  "reanalysis_precip_amt_kg_per_m2","reanalysis_sat_precip_amt_mm",
                  "station_avg_temp_c","station_max_temp_c","station_min_temp_c",
                  "station_precip_mm","ndvi_nw","ndvi_se","ndvi_sw",
                  "weekofyear", "week_start_date","year","Lag1", "Lag2","Lag3","Lag4","Lag5"]
output_features = ["total_cases", "week_start_date","year"]

# Removing the NAs produced due to Lags
sj_preprocessed = sj_train_processed.dropna()
iq_preprocessed = iq_train_processed.dropna()

## Predictors and Targets for building the model

sj_model_train,sj_model_val = train_test_split(sj_preprocessed, test_size=0.2)
iq_model_train,iq_model_val = train_test_split(iq_preprocessed, test_size=0.2)

X_train_sj = sj_model_train.drop(output_features,axis=1).values

y_train_sj = sj_model_train.drop(input_features,axis=1).values

X_train_iq = iq_model_train.drop(output_features,axis=1).values

y_train_iq = iq_model_train.drop(input_features,axis=1).values


## Gradient boosting Regression

regressor_gb_sj = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth = 30)

regressor_gb_sj.fit(X_train_sj,y_train_sj.ravel())
regressor_gb_iq = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth = 30)

regressor_gb_iq.fit(X_train_iq,y_train_iq.ravel())



Output_total_Cases = []
for i in range(len(sj_test_data_processed.index)):
    sj_train_processed = pd.concat([sj_train_processed, sj_test_data_processed.iloc[[i]]])
    sj_train_processed.at[sj_train_processed.index[-1],"Lag1"] = sj_train_processed.iloc[-2]["total_cases"]
    sj_train_processed.at[sj_train_processed.index[-1],"Lag2"] = sj_train_processed.iloc[-3]["total_cases"]
    sj_train_processed.at[sj_train_processed.index[-1],"Lag3"] = sj_train_processed.iloc[-4]["total_cases"]
    sj_train_processed.at[sj_train_processed.index[-1],"Lag4"] = sj_train_processed.iloc[-5]["total_cases"]
    sj_train_processed.at[sj_train_processed.index[-1],"Lag5"] = sj_train_processed.iloc[-6]["total_cases"]

    y=sj_train_processed.drop(output_features, axis=1).values
    k = regressor_gb_sj.predict(numpy.array(y[-1]).reshape(1,-1))
    sj_train_processed.at[sj_train_processed.index[-1],"total_cases"] = int(k)
    Output_total_Cases.append(sj_train_processed.iloc[-1]["total_cases"])

for i in range(len(iq_test_data_processed.index)):
    iq_train_processed = pd.concat([iq_train_processed, iq_test_data_processed.iloc[[i]]])
    iq_train_processed.at[iq_train_processed.index[-1],"Lag1"] = iq_train_processed.iloc[-2]["total_cases"]
    iq_train_processed.at[iq_train_processed.index[-1],"Lag2"] = iq_train_processed.iloc[-3]["total_cases"]
    iq_train_processed.at[iq_train_processed.index[-1],"Lag3"] = iq_train_processed.iloc[-4]["total_cases"]
    iq_train_processed.at[iq_train_processed.index[-1],"Lag4"] = iq_train_processed.iloc[-5]["total_cases"]
    iq_train_processed.at[iq_train_processed.index[-1],"Lag5"] = iq_train_processed.iloc[-6]["total_cases"]

    y=iq_train_processed.drop(output_features, axis=1).values
    k = regressor_gb_iq.predict(numpy.array(y[-1]).reshape(1,-1))
    iq_train_processed.at[iq_train_processed.index[-1],"total_cases"] = int(k)
    Output_total_Cases.append(iq_train_processed.iloc[-1]["total_cases"])



csv_output = pd.read_csv("submission_format.csv")
csv_output['total_cases'] = Output_total_Cases
# print(csv_output.head())
csv_output.to_csv('submission_output.csv')
