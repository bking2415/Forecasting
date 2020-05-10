# Intial code was written Jupyter Notebook
# then edited to run using PyCharm
# Referenced from karamanbk/g6_intro.py
from __future__ import division

from datetime import datetime, timedelta, date
import os

import inline as inline
import matplotlib
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import jupyter

import warnings

warnings.filterwarnings("ignore")

# Import Plotly
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.offline as pyoff

# import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# initiate plotly
# pyoff.init_notebook_mode()

# get path to the data
os.chdir(os.path.dirname(__file__))
path = os.path.join(os.getcwd(), "demand-forecasting-kernels-only")

file = "train.csv"

# read the data in csv
df_sales = pd.read_csv(path + "/" + file)

# convert date field from string to datetime
df_sales['date'] = pd.to_datetime(df_sales['date'])

# show first 10 rows
# print(df_sales.head(10))

# Our task is to forecast monthly total sales.
# We need to aggregate our data at the monthly level
# and sum up the sales column.

# represent month in date field as its first day
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])

# groupby date and sum the sales
df_sales = df_sales.groupby('date').sales.sum().reset_index()

# print(df_sales.head())

# plot monthly sales
plot_data = [
    go.Scatter(
        x=df_sales['date'],
        y=df_sales['sales'],
    )
]
plot_layout = go.Layout(
    title='Montly Sales'
)
fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)
# Plot monthly sales
# pyoff.plot(fig, filename='monthlysales.html')

# Obviously, it is not stationary and
# has an increasing trend over the months.
# One method is to get the difference in sales
# compared to the previous month and build the model on it:

# Create a new dataframe to model the difference
df_diff = df_sales.copy()
# Add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)
# Drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])
# print(df_diff.head(10))

# plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['date'],
        y=df_diff['diff'],
    )
]
plot_layout = go.Layout(
    title='Monthly Sales Diff'
)
fig = go.Figure(data=plot_data, layout=plot_layout)

# Plot monthly difference sales
# pyoff.plot(fig, filename='monthlydiffsales.html')
# pyoff.plot(fig)

# create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'], axis=1)
# adding lags
for inc in range(1, 13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
# drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

lagList = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'lag_11',
           'lag_12']
# sum over the column axis.
# df_supervised['total'] = df_supervised.loc[:, lagList].sum(axis=1)

print(df_supervised.head(5))

# Adjusted R-squared is the answer.
# tells us how good our features explain the variation in our label
# The higher variation the better fit

# # Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + '
                        'lag_11 + lag_12', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

# import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler

df_model = df_supervised.drop(['sales', 'date'], axis=1)
# split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

# apply Min Max Scaler
# Scale variables of train and test
# Normalized the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

# Build LSTM Model

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test, batch_size=1)
# for multistep prediction, you need to replace X_test values with the predictions coming from t-1

# reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
# rebuild test set for inverse transform
pred_test_set = []
for index in range(0, len(y_pred)):
    print(np.concatenate([y_pred[index], X_test[index]], axis=1))
    pred_test_set.append(np.concatenate([y_pred[index], X_test[index]], axis=1))
# reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
# inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

# create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:].date)
act_sales = list(df_sales[-7:].sales)
for index in range(0, len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index + 1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
# for multistep prediction, replace act_sales with the predicted sales
# print(df_result)

# merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales, df_result, on='date', how='left')
# print("Sales Prediction")
# print(df_sales_pred['date'])
# print(df_sales_pred['sales'])
# print(df_sales_pred['pred_value'])
# Interactive Plot
# plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['sales'],
        name='actual'
    ),
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )

]
plot_layout = go.Layout(
    title='Sales Prediction'
)
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig, filename='salesPredictions.html')

# Alternative plot in .png file
x1 = df_sales_pred['date']
# Data for the first line
y1 = df_sales_pred['sales']

x2 = df_sales_pred['date']
# Data for the second line
y2 = df_sales_pred['pred_value']

plt.plot(x1, y1)
plt.plot(x2, y2)

plt.legend(["actual", "predicted"])
plt.show()
