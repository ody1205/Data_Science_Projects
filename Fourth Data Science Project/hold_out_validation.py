# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:57:33 2019

@author: ody12
"""

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

dc_listings = dc_listings.drop(['room_type', 'city', 'state', 'latitude', 
                                'longitude', 'zipcode', 'host_acceptance_rate',
                                'host_listings_count', 'host_response_rate', 
                                'cleaning_fee', 'security_deposit'], 
                                axis = 1)
dc_listings = dc_listings.dropna(axis = 0)

stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)

#split_one = dc_listings.iloc[:1862].copy()
#split_two = dc_listings.iloc[1862:].copy()

#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.metrics import mean_squared_error
#
#train_one = split_one
#test_one = split_two
#train_two = split_two
#test_two = split_one
#
#knn = KNeighborsRegressor()
#knn.fit(train_one[['accommodates']], train_one['price'])
#predictions = knn.predict(test_one[['accommodates']])
#one_mse = mean_squared_error(predictions, test_one['price'])
#iteration_one_rmse = one_mse ** (1/2)
#
#knn.fit(train_two[['accommodates']], train_two['price'])
#predictions = knn.predict(test_two[['accommodates']])
#two_mse = mean_squared_error(predictions, test_two['price'])
#iteration_two_rmse = two_mse ** (1/2)
#
#avg_rmse = np.mean([iteration_one_rmse, iteration_two_rmse])
#
#dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
#dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
#dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
#dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
#dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5
#
#print(dc_listings['fold'].value_counts())
#print(dc_listings['fold'].isnull().sum())
#
##hold out fold number one and set it to a test set.

#from sklearn.metrics import mean_squared_error


#train_iteration_one = dc_listings[dc_listings['fold'] != 1]
#test_iteration_one = dc_listings[dc_listings['fold'] == 1].copy()
#knn.fit(train_iteration_one[['accommodates']], train_iteration_one['price'])
#predictions = knn.predict(test_iteration_one[['accommodates']])
#iteration_one_mse = mean_squared_error(predictions, test_iteration_one['price'])
#iteration_one_rmse = iteration_one_mse ** (1/2)
#
## Use np.mean to calculate the mean.
#fold_ids = [1,2,3,4,5]
#
#def train_and_validate(df, folds):
#    rmses = []
#    for i in folds:
#        train = df[df['fold'] != i]
#        test = df[df['fold'] == i]         
#        knn = KNeighborsRegressor()
#        knn.fit(train[['accommodates']], train['price'])
#        predictions = knn.predict(test[['accommodates']])
#        mse = mean_squared_error(predictions, test['price'])
#        rmses.append(mse ** (1/2))
#    return rmses
#
#rmses = train_and_validate(dc_listings, fold_ids)
#avg_rmse = np.mean(rmses)
#print(rmses)
#print(avg_rmse)
##print('predicted price',predictions)
#
##using sklearn instead the function I wrote.
#from sklearn.model_selection import cross_val_score, KFold
#
#kf = KFold(5, shuffle = True, random_state = 1)
#knn = KNeighborsRegressor()
#
#mses = cross_val_score(knn, dc_listings[['accommodates']], dc_listings['price'], scoring = 'neg_mean_squared_error', cv = kf)
#rmses = np.sqrt(np.absolute(mses))
#avg_rmse = np.mean(rmses)
#
#print(rmses)
#print(avg_rmse)
train_df = dc_listings.iloc[:1862].copy()
test_df = dc_listings.iloc[1862:].copy()
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]
avg_rmses = []
train_columns = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
    
for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    knn = KNeighborsRegressor(n_neighbors = 6, algorithm = 'brute')
    knn.fit(train_df[train_columns], train_df['price'])
    predictions = knn.predict(test_df[train_columns])
    mses = cross_val_score(knn, dc_listings[train_columns], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))
    avg_rmses.append(avg_rmse)
plt.scatter(num_folds, avg_rmses)
plt.xlabel('Number of Folds')
plt.ylabel('Average RMSE')
plt.show()

print('AVG RMSE across 3 ~ 23 folds: ', np.mean(avg_rmses))
