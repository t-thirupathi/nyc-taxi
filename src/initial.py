import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Train on lower number of points if running on laptop
#train = train.head(30000)

# Convert to datetime object to easily extract hour of day, day of week
train['pickup_datetime_object'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime_object'] = pd.to_datetime(test['pickup_datetime'])

# Day of week and hour of day affects the ride duration most
train['day_of_week'] = train['pickup_datetime_object'].apply(lambda x: x.weekday())
test['day_of_week'] = test['pickup_datetime_object'].apply(lambda x: x.weekday())
train['hour_of_day'] = train['pickup_datetime_object'].apply(lambda x: x.hour)
test['hour_of_day'] = test['pickup_datetime_object'].apply(lambda x: x.hour)

# Convert day of week and hour of day to cyclic encoding
# Otherwise day 0 (Sunday) and day 6 (Saturday) are far apart, which is wrong
def day_of_week_sine(x):
    return np.sin(2 * np.pi * x / 7)

train['day_of_week_sine'] = train['day_of_week'].apply(day_of_week_sine)
test['day_of_week_sine'] = test['day_of_week'].apply(day_of_week_sine)

def hour_of_day_sine(x):
    return np.sin(2 * np.pi * x / 24)

train['hour_of_day_sine'] = train['hour_of_day'].apply(hour_of_day_sine)
test['hour_of_day_sine'] = test['hour_of_day'].apply(hour_of_day_sine)

# Compute the aerial distance. 
# Not considering the curvature of earth, 
# since all data points are localized to a very small area - New York city
train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']
train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['distance'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['distance'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))

# View the correlation between each other variables
print(train.corr())

# Select only few variables as input, based on correlation analysis
x_train = train[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]
y_train = train['trip_duration']
x_test = test[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]

# Since number of data points is huge, 
# use linear regression model with low tolerance, and saga solver
# otherwise it takes to months to train. 
model = LogisticRegression(tol=0.1, solver='saga')
model.fit(x_train, y_train)

test['trip_duration'] = model.predict(x_test)

# We need to submit only the id and predicted trip duration
test[['id', 'trip_duration']].to_csv('submission.csv', index=False)

