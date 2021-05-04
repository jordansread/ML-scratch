import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import numpy as np

train = read_csv('data/train_data_predict_temperature.csv')
test = read_csv('data/test_data_predict_temperature.csv')

# create a lagged predictor for solar radiation (I read that may be important):
sol_lagged = pd.Series([1]).append(train['solar_radiation'].head(-1), ignore_index = True)

# insert the new variable into the third column
train.iloc[:,2] = sol_lagged

x_train = train.iloc[:, 2:]
x_test = test.iloc[:, 2:]
y_train = train['water_temperature']
y_test = test['water_temperature']

model_xgb = xgb.XGBRegressor(n_estimators = 200,
                             objective = 'reg:squarederror',
                             max_depth = 4,
                             booster = 'gbtree',
                             learning_rate = 0.5,
                             gamma = 0,
                             min_child_weight = 1,
                             subsample = 1,
                             n_jobs = 8)
model_xgb.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              verbose=False)

preds = model_xgb.predict(x_test)
obs = y_test

plt.plot(preds, obs, 'o', color = 'black')
plt.xlabel('observed temperature (°C)')
plt.ylabel('predicted temperature (°C)')
axes = plt.gca()
x_line = np.arange(axes.get_xlim()[0],axes.get_xlim()[1])
plt.plot(x_line, x_line, '-', color = 'black')
