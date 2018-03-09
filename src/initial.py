
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression


# In[91]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['pickup_datetime'] = train['pickup_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
test['pickup_datetime'] = test['pickup_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


# In[92]:


train = train.head(1000)
test = test.head(1000)


# In[93]:


train['day_of_week'] = train['pickup_datetime'].apply(lambda x: x.weekday())
test['day_of_week'] = test['pickup_datetime'].apply(lambda x: x.weekday())
train['hour_of_day'] = train['pickup_datetime'].apply(lambda x: x.hour)
test['hour_of_day'] = test['pickup_datetime'].apply(lambda x: x.hour)


# In[95]:


def day_of_week_sine(x):
    return np.sin(2 * np.pi * x / 7)

train['day_of_week_sine'] = train['day_of_week'].apply(day_of_week_sine)
test['day_of_week_sine'] = test['day_of_week'].apply(day_of_week_sine)

def hour_of_day_sine(x):
    return np.sin(2 * np.pi * x / 24)

train['hour_of_day_sine'] = train['hour_of_day'].apply(hour_of_day_sine)
test['hour_of_day_sine'] = test['hour_of_day'].apply(hour_of_day_sine)


# In[97]:


def distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    return np.sqrt((dropoff_latitude - pickup_latitude) ** 2 + (dropoff_longitude - pickup_longitude) ** 2)

train['distance'] = train.apply(lambda x: distance(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
test['distance'] = test.apply(lambda x: distance(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)


# In[98]:


train_tmp = train[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine', 'trip_duration']]
print train_tmp.corr()


# In[100]:


x_train = train[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]
y_train = train['trip_duration']
x_test = test[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]


# In[ ]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[89]:


test['trip_duration'] = model.predict(x_test)
test.to_csv('submission.csv', index=False)

