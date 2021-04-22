
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from os.path import split
import pandas as pd
import numpy as np
from defrost_data_creator import *
from functions import *
from nn_model import *
import math
from sklearn.metrics import mean_squared_error

import tensorflow as tf

import matplotlib.pyplot as plt

 
# read the BSH data
df = pd.read_csv("SDR30EU-32C-Data-mod.csv", encoding="utf-16")

# get the defrsot heater column to prepare the label
defrost  = df["FgDefrHeater"]

# drop the label column and other columns from the main dataset
df = df.drop(["FgDefrHeater", "Time", "AT", "AH"], axis=1)
print(len(list(df)))



# get the start and end indexs(timesteps) of each defrost interval in the defrsot column.
defrsot_intervals = get_defrsot_intervals(defrost)
print("num of defrost intervals, ",len(defrsot_intervals))



# get the intervals between the defrostings (used to estimate the defrosting)
intervals_between_defrosts = get_intervals_between_defrosts(dataset, defrsot_intervals)
print(np.asarray(intervals_between_defrosts).shape)





# get the intervals between the defrostings (these intervals are used to estimate the defrosting level)
intervals_between_defrosts = get_intervals_between_defrosts(dataset, defrsot_intervals)
print(np.asarray(intervals_between_defrosts).shape)




# the type is the length of each defrsot interval ( here it is 3 to 18)
types = [(x[1] - x[0]) for x in defrsot_intervals]
types = np.asarray(types) // 10


# create a labeled dataset consisting of interval-label pair
intervals_with_types = []
for i,interval in enumerate(intervals_between_defrosts):
    intervals_with_types.append([interval, types[i]])



# intervals are of different lengths, we need to unify them, set a length discarding any intervals smaller then it
length = 15
intervals_with_types = [x for x in intervals_with_types if len(x[0]) > length]

# unify the intervals, crop earlier records
new_intervals = []
for interv in intervals_with_types:
    new_intervals.append([interv[0][-length:], interv[1]])


print(len(new_intervals))

# intervals_with_types = intervals_with_types.reshape((89,20, 68, 1))


# split the data into train-test
split_at = int(0.50 * len(new_intervals))
train  = new_intervals[:split_at]
test = new_intervals[split_at:]

train, scaler = standardize(dataset)
test = standardize_with_scaler(test, scaler)

#separate data and labels
train_data = [x[0] for x in train]
lstm_train_labels = [x[1] for x in train]

test_data = [x[0] for x in test]
lstm_test_labels = [x[1] for x in test]



train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
# print(test_data)
# print(lstm_test_labels)




#model path to be saved
path = "model.h5"
lstm_train_labels = np.asarray(lstm_train_labels)
lstm_test_labels = np.asarray(lstm_test_labels)


print("train image shape ",train_data.shape)
print("train labels ",lstm_train_labels)


# create the lstm model and train it.
model =  create_mlp_model(train_data.shape[1], train_data.shape[2])

print(model.summary())
train_model(model, train_data, lstm_train_labels, path, epochs=100, batch_size=64)


# model = tf.keras.models.load_model(path)


# make predictions
trainPredict = model.predict(train_data)
testPredict = model.predict(test_data)


print(lstm_train_labels)
print(trainPredict)
print(lstm_test_labels)
print(testPredict)


testScore = math.sqrt(mean_squared_error(lstm_test_labels, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


#print results
real_predicted  = []
for i in range(len(testPredict)):
    real_predicted.append([lstm_test_labels[i], testPredict[i]])


plt.plot(real_predicted)
plt.legend(["test_samples", "predicted"], bbox_to_anchor=(0.75, 1.15), ncol=2)
plt.show()

