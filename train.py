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
from matplotlib import interactive
## Reading the Input Data
input_data = pd.read_csv("dengue_features_train.csv")
label_data = pd.read_csv("dengue_labels_train.csv")

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

## Partitioning the dataset based on cities
sj_train = merged_train.loc[merged_train["city"] == "sj"]

iq_train = merged_train.loc[merged_train["city"] == "iq"]
# print("The shape of total dataset is as follows")
# print(merged_train.shape)
# print("The shape of sj dataset is as follows")
# print(sj_train.shape)
# print("The shape of iq dataset is as follows")
# print(iq_train.shape)

## Attributes usually follow the most recent trends, so replacing the NA values with most recent values.
sj_train.fillna(method='ffill',inplace=True)
iq_train.fillna(method='ffill',inplace=True)
sj_train = sj_train.dropna()
iq_train = iq_train.dropna()
# print(sj_train_drop.shape)
# print(iq_train_drop.shape)

## Converting the Week Start Date to a time series object
sj_train['week_start_date'] = pd.to_datetime(sj_train.week_start_date)
sj_train['week_start_date'] = sj_train['week_start_date'].dt.strftime('%d-%m-%Y')
iq_train['week_start_date'] = pd.to_datetime(iq_train.week_start_date)
iq_train['week_start_date'] = iq_train['week_start_date'].dt.strftime('%d-%m-%Y')
#print(sj_train['week_start_date'])
#print(iq_train['week_start_date'])

##Plotting a graph which gives the information about the start date and total cases for that week
# plt.plot(sj_train.week_start_date, sj_train.total_cases, color='g')
# plt.plot(iq_train.week_start_date, iq_train.total_cases, color='orange')
# plt.xlabel('Week start date')
# plt.ylabel('Total cases')
# plt.title('Dengue cases in a week ')

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
# print(regr.feature_importances_)
# print(sj_reg_col)
# plt.plot(sj_reg_col,regr.feature_importances_,'ro')


#Plotting the Correlation between features
cat_attr = ["weekofyear", "city", "week_start_date","year"]
correlations = sj_train.drop(cat_attr,axis=1).corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# names = [ 'ndvi_ne', 'ndvi_nw' ,'ndvi_se', 'ndvi_sw'
#  'precipitation_amt_mm', 'reanalysis_air_temp_k' ,'reanalysis_avg_temp_k'
#  'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
#  'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
#  'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
#  'reanalysis_specific_humidity_g_per_kg' ,'reanalysis_tdtr_k',
#  'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
#  'station_min_temp_c' ,'station_precip_mm', 'total_cases']
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = numpy.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# print(correlations)

## Removing the unwanted features/ features with high correlation
unwanted_features = ['city','reanalysis_tdtr_k','reanalysis_relative_humidity_percent',
                 'reanalysis_specific_humidity_g_per_kg','ndvi_ne','station_diur_temp_rng_c']
sj_train_processed = sj_train.drop(unwanted_features,axis=1)
iq_train_processed = iq_train.drop(unwanted_features,axis=1)

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

X_val_sj = sj_model_val.drop(output_features,axis=1).values

y_val_sj = sj_model_val.drop(input_features,axis=1).values

X_val_iq = iq_model_val.drop(output_features,axis=1).values

y_val_iq = iq_model_val.drop(input_features,axis=1).values

## Random forest Regression

regressor_sj = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=2000)
regressor_sj.fit(X_train_sj,y_train_sj.ravel())
regressor_iq = RandomForestRegressor(max_depth=5, random_state=10, n_estimators=2000)
regressor_iq.fit(X_train_iq,y_train_iq.ravel())

predictions_sj = regressor_sj.predict(X_val_sj)
predictions_iq = regressor_iq.predict(X_val_iq)

## Performance Metrics for Trained Model using Random Forest
print("------- Performance metrics for Random forest SJ Model -----")
print("R square score for SJ City predictions" + str(r2_score(y_val_sj,predictions_sj)))
print(" Mean Absolute Error for SJ City predictions" + str(mean_absolute_error(y_val_sj,predictions_sj)))

print("###############################################")
print("------- Performance metrics for Random forest IQ Model -----")
print("R square score for IQ City predictions" + str(r2_score(y_val_iq,predictions_iq)))
print(" Mean Absolute Error for IQ City predictions" + str(mean_absolute_error(y_val_iq,predictions_iq)))


## K NEAREST NEIGHBOURS REGRESSION
knn_sj_regressor = neighbors.KNeighborsRegressor(5, weights='distance')
knn_sj_regressor.fit(X_train_sj,y_train_sj.ravel())
knn_iq_regressor = neighbors.KNeighborsRegressor(5, weights='distance')
knn_iq_regressor.fit(X_train_iq,y_train_iq.ravel())




knn_predictions_sj = knn_sj_regressor.predict(X_val_sj)
knn_predictions_iq = knn_iq_regressor.predict(X_val_iq)

## Performance Metrics for Trained Model using KNN
print("------- Performance metrics  for k nearest neighbours SJ Model -----")
print("R square score for SJ City predictions" + str(r2_score(y_val_sj,knn_predictions_sj)))
print(" Mean Absolute Error for SJ City predictions" + str(mean_absolute_error(y_val_sj,knn_predictions_sj)))

print("###############################################")
print("------- Performance metrics for k nearest neighbours IQ Model -----")
print("R square score for IQ City predictions" + str(r2_score(y_val_iq,knn_predictions_iq)))
print(" Mean Absolute Error for IQ City predictions" + str(mean_absolute_error(y_val_iq,knn_predictions_iq)))


## bayesan ridge regression
reg = linear_model.BayesianRidge()
bayesan_sj =  reg.fit(X_train_sj,y_train_sj.ravel())
bayesan_iq =  reg.fit(X_train_iq,y_train_iq.ravel())
bayesan_predictions_sj= bayesan_sj.predict(X_val_sj)
bayesan_predictions_iq = bayesan_iq.predict(X_val_iq)

## Performance Metrics for Trained Model using Bayesan Ridge regression
print("------- Performance metrics  for Bayesan Ridge regression SJ Model -----")
print("R square score for SJ City predictions" + str(r2_score(y_val_sj,bayesan_predictions_sj)))
print(" Mean Absolute Error for SJ City predictions" + str(mean_absolute_error(y_val_sj,bayesan_predictions_sj)))

print("###############################################")
print("------- Performance metrics for Bayesan Ridge regression IQ Model -----")
print("R square score for IQ City predictions" + str(r2_score(y_val_iq,bayesan_predictions_iq)))
print(" Mean Absolute Error for IQ City predictions" + str(mean_absolute_error(y_val_iq,bayesan_predictions_iq)))

##support vector regression
svr_rbf = SVR(kernel='poly', C=1e3, gamma=0.2)
svr_sj =  reg.fit(X_train_sj,y_train_sj.ravel())
svr_iq =  reg.fit(X_train_iq,y_train_iq.ravel())
svr_predictions_sj= svr_sj.predict(X_val_sj)
svr_predictions_iq = svr_iq.predict(X_val_iq)


## Performance Metrics for Trained Model using Support Vector regression
print("------- Performance metrics for Support Vector Regression SJ Model -----")
print("R square score for SJ City predictions" + str(r2_score(y_val_sj,svr_predictions_sj)))
print(" Mean Absolute Error for SJ City predictions" + str(mean_absolute_error(y_val_sj,svr_predictions_sj)))

print("###############################################")
print("------- Performance metrics for Support Vector Regression IQ Model -----")
print("R square score for IQ City predictions" + str(r2_score(y_val_iq,svr_predictions_iq)))
print(" Mean Absolute Error for IQ City predictions" + str(mean_absolute_error(y_val_iq,svr_predictions_iq)))

###Gradient boosting Regression
regressor_gb_sj = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth = 20)

regressor_gb_sj.fit(X_train_sj,y_train_sj.ravel())
gb_predictions_sj= regressor_gb_sj.predict(X_val_sj)
regressor_gb_iq = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth = 20)

regressor_gb_iq.fit(X_train_iq,y_train_iq.ravel())
gb_predictions_iq = regressor_gb_iq.predict(X_val_iq)


###Performance Metrics for Trained Model using Gradient Boosting Regression

print("------- Performance metrics for gradient boosting Regression SJ Model -----")
print("R square score for SJ City predictions" + str(r2_score(y_val_sj,gb_predictions_sj)))
print(" Mean Absolute Error for SJ City predictions" + str(mean_absolute_error(y_val_sj,gb_predictions_sj)))

print("###############################################")
print("------- Performance metrics for gradient boosting Regression IQ Model -----")
print("R square score for IQ City predictions" + str(r2_score(y_val_iq,gb_predictions_iq)))
print(" Mean Absolute Error for IQ City predictions" + str(mean_absolute_error(y_val_iq,gb_predictions_iq)))

# plt.plot(y_val_iq,'green', gb_predictions_iq, 'red')
#
# plt.xlabel('Actual Dengue cases of city IQUITOS in Week')
# plt.ylabel('Predicted Dengue cases of city IQUITOS in Week ')
# plt.title('Dengue cases in a week ')

plt.plot(y_val_sj,'green', gb_predictions_sj,'red')
plt.xlabel('Actual Dengue cases  of city San Juan in Week')
plt.ylabel('Predicted Dengue cases of San Juan in Week ')
plt.title('Dengue cases in a week ')

plt.show()
