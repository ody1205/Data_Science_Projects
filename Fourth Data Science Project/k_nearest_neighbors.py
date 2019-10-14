# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 01:38:48 2019

@author: Min Oh
"""

import pandas as pd
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.head(0))

first_living_space_value = dc_listings.iloc[0]['accommodates']
first_distance = np.abs(first_living_space_value - 3)
print(first_distance)

#create distance column to find nearing data sets
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - 3))
print(dc_listings['distance'].value_counts)

#eliminates bias by data roder
np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.iloc[0:10]['price'])

no_comma = dc_listings['price'].str.replace(',', '')
no_dollar = no_comma.str.replace('$', '')
dc_listings['price'] = no_dollar.astype('float')

#for 3 people per night price
mean_price = dc_listings.iloc[:5]['price'].mean()
print(mean_price)


#k_nearest_neighbors function with k = 5
def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    predict_price = temp_df.iloc[:5]['price'].mean()
    return predict_price


acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)

print(acc_one)
print(acc_two)
print(acc_four)


#evaluating model with a training set(75%) and a test set(25%) of the data
train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df['accommodates'].apply(predict_price)

#error metric 
#MAE mean absolute error 
#(how far off the prediction is in either the positive or negative direction.)
test_df['error'] = np.absolute(test_df['predicted_price'] - test_df['price'])
mae = test_df['error'].mean()
print(mae)
#MSE mean squred error (make the error values clear to observe)
test_df['squred_error'] = (test_df['predicted_price'] - test_df['price'])**2
mse = test_df['squred_error'].mean()
print(mse)
#RMSE root mean squared error (base unit)
rmse = mse ** (1/2)
print(rmse)

#This can be done in this way as well

errors_one = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 
                        10, 5, 10, 5, 10, 5, 10])
errors_two = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 
                        10, 5, 10, 5, 10, 5, 1000])

mae_one = errors_one.sum()/len(errors_one)
rmse_one = ((errors_one**2).sum()/len(errors_one))**(1/2)

mae_two = errors_two.sum()/len(errors_two)
rmse_two = ((errors_two**2).sum()/len(errors_two))**(1/2)
'''
improving the k-nearest model by increasing the number of attributes the model
uses to calculate similarity when ranking the closest neighbors
1. non-numerical values (e.g. city or state)
Euclidean distance equation expects numerical values
2. missing values
distance equation expects a value for each observation and attribute
3. non-ordinal values (e.g. latitude or longitude)
ranking by Euclidean distance doesn't make sense if all attributes aren't 
ordinal
'''

dc_listings.info()
'''
room-tyep, city, state contains non-numerical values
latitude, longitude, zipcode contain non-ordinal values
host_response_rate, host_acceptance_rate, host_listings_count cannot be used
as it is hard to group living sWpaces to the hosts themselves
'''

#remove these columns
dc_listings = dc_listings.drop(['room_type', 'city', 'state', 'latitude', 
                                'longitude', 'zipcode', 'host_acceptance_rate',
                                'host_listings_count', 'host_response_rate'], 
                                axis = 1)

#cleaning_fee cloumn has 37.3% rows missing and 
#security_deposit coulmn has 61.7% rows missing. It is best to remove them.
#for the rest of the remaining columns, delete missing rows.
dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis = 1)
dc_listings = dc_listings.dropna(axis = 0)
dc_listings.info()

#normalize the listings by using x = x - x.mean() / x.std
normalized_listings = (dc_listings - dc_listings.mean())/(dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))

#using two attributes from the normalized listing (accomodates and bathrooms)
#to calculate euclidean distance. First try first and fifth rows.
from scipy.spatial import distance

first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates', 'bathrooms']]

first_fifth_distance = distance.euclidean(first_listing, fifth_listing)

print(first_fifth_distance)


#learning sklearn
from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]
train_columns = ['accommodates', 'bathrooms']

knn = KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')
train_target = train_df['price']
knn.fit(train_df[train_columns], train_target)

predictions = knn.predict(test_df[train_columns])

#mse and rmse
from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])

two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = two_features_mse ** (1/2)
print(two_features_mse)
print(two_features_rmse)

#using four features
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

knn.fit(train_df[features], train_df['price'])
four_predictions = knn.predict(test_df[features])

four_mse = mean_squared_error(four_predictions, test_df['price'])
four_rmse = four_mse ** (1/2)
print(four_mse)
print(four_rmse)

train_columns = train_df.columns.tolist()
train_columns.remove('price')


#using all the features
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')
knn.fit(train_df[train_columns], train_df['price'])
all_features_predictions = knn.predict(test_df[train_columns])

from sklearn.metrics import mean_squared_error

all_features_mse = mean_squared_error(all_features_predictions, test_df['price'])
all_features_rmse = all_features_mse ** (1/2)
print(all_features_mse)
print(all_features_rmse)

'''
The RMSE value increased to 125.1 when using all the features.
Using more features does not guarentee imporve prediction accuracy.
We will need to select the relevant attributes the model uses 
to calculate similarity when ranking the closest neighbors. (feature selection)
'''

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
#import new test and training sets.
import pandas as pd

train_df = pd.read_csv('dc_airbnb_train.csv')
test_df = pd.read_csv('dc_airbnb_test.csv')

#try various k values (1 - 20)
hyper_params = [i for i in range(1, 21)]
    
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

mse_values = []
train_columns = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
for i in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = i, algorithm = 'brute')
    knn.fit(train_df[train_columns], train_df['price'])
    predictions = knn.predict(test_df[train_columns])
    mse_values.append(mean_squared_error(predictions, test_df['price']))
print(mse_values)

#it seems like 6 is the most optimal k-value.
plt.scatter(hyper_params, mse_values)
plt.show()

#using all the features
hyper_params = [x for x in range(1,21)]
mse_values = list()

train_columns = train_df.columns.tolist()
train_columns.remove('price')
for i in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = i, algorithm = 'brute')
    knn.fit(train_df[train_columns], train_df['price'])
    predictions = knn.predict(test_df[train_columns])
    mse_values.append(mean_squared_error(predictions, test_df['price']))
    
    
#seems like 11 is the best k-value.    
plt.scatter(hyper_params, mse_values)
plt.show()

#try to return the lowest values
two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
# Append the second model's MSE values to this list.
three_mse_values = list()
two_hyp_mse = dict()
three_hyp_mse = dict()

for i in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = i, algorithm = 'brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    two_mse_values.append(mean_squared_error(predictions, test_df['price']))

plt.scatter(hyper_params, two_mse_values)
plt.show()

two_lowest_mse = two_mse_values[0]
two_lowest_k = 1

for k, mse in enumerate(two_mse_values):
    if mse < two_lowest_mse:
        two_lowest_mse = mse
        two_lowest_k = k + 1
        
two_hyp_mse[two_lowest_k] = two_lowest_mse

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])
    mse = mean_squared_error(test_df['price'], predictions)
    three_mse_values.append(mse)

plt.scatter(hyper_params, three_mse_values)
plt.show()

three_lowest_mse = three_mse_values[0]
three_lowest_k = 1

for k,mse in enumerate(three_mse_values):
    if mse < three_lowest_mse:
        three_lowest_mse = mse
        three_lowest_k = k + 1
        
three_hyp_mse[three_lowest_k] = three_lowest_mse

print(two_hyp_mse)
print(three_hyp_mse)

'''Holdout validation technique
splitting the full dataset into 2 partitions: a training set ,a test set.
training the model on the training set,
using the trained model to predict labels on the test set,
computing an error metric to understand the model's effectiveness,
switch the training and test sets and repeat, average the errors.
In holdout validation, we usually use a 50/50 split instead of the 75/25 split
from train/test validation. This way, we remove the number of observations as
a potential source of variation in our model performance.
'''

import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)

split_one = dc_listings.iloc[:1862].copy()
split_two = dc_listings.iloc[1862:].copy()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

knn = KNeighborsRegressor()
knn.fit(train_one[['accommodates']], train_one['price'])
predictions = knn.predict(test_one[['accommodates']])
one_mse = mean_squared_error(predictions, test_one['price'])
iteration_one_rmse = one_mse ** (1/2)

knn.fit(train_two[['accommodates']], train_two['price'])
predictions = knn.predict(test_two[['accommodates']])
two_mse = mean_squared_error(predictions, test_two['price'])
iteration_two_rmse = two_mse ** (1/2)

avg_rmse = np.mean([iteration_one_rmse, iteration_two_rmse])

dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

print(dc_listings['fold'].value_counts())
print(dc_listings['fold'].isnull().sum())

#hold out fold number one and set it to a test set.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor()
train_iteration_one = dc_listings[dc_listings['fold'] != 1]
test_iteration_one = dc_listings[dc_listings['fold'] == 1].copy()
knn.fit(train_iteration_one[['accommodates']], train_iteration_one['price'])
predictions = knn.predict(test_iteration_one[['accommodates']])
iteration_one_mse = mean_squared_error(predictions, test_iteration_one['price'])
iteration_one_rmse = iteration_one_mse ** (1/2)

# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]

def train_and_validate(df, folds):
    rmses = []
    for i in folds:
        train = df[df['fold'] != i]
        test = df[df['fold'] == i]         
        knn = KNeighborsRegressor()
        knn.fit(train[['accommodates']], train['price'])
        predictions = knn.predict(test[['accommodates']])
        mse = mean_squared_error(predictions, test['price'])
        rmses.append(mse ** (1/2))
    return rmses

rmses = train_and_validate(dc_listings, fold_ids)
avg_rmse = np.mean(rmses)
print(rmses)
print(avg_rmse)

#using sklearn instead the function I wrote.
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(5, shuffle = True, random_state = 1)
knn = KNeighborsRegressor()

mses = cross_val_score(knn, dc_listings[['accommodates']], dc_listings['price'], scoring = 'neg_mean_squared_error', cv = kf)
rmses = np.sqrt(np.absolute(mses))
avg_rmse = np.mean(rmses)

print(rmses)
print(avg_rmse)

