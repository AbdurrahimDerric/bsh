from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
import random
from keras import optimizers, Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

df = pd.read_csv("fridge_data.csv")
feature_names = list(df)
print(feature_names)

door_plot = df["DoorCLOSED"] * 100
humidity = df["InHumid"]
temp = df["inTemp"]

# plt.plot(door_plot)
# plt.plot(humidity)
# plt.plot(temp)
# plt.legend(["door", "humidity","temp"])
# plt.show()

# exit()




# take the first fouth features to be independent varibales
X_df = df[feature_names[:4]] 
# get the numpy array of the faetures
X = X_df.values 

# take the door state as a label to be predicted. the door feature is already encoded as 0: closed, 1:open
y_df = df["DoorCLOSED"]
y = y_df.values
y = y.reshape(-1,1)
print(len(X))
print(len(y))
# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y) 


# split according to the length of the dataset, hint: x = x[:split_at]
split_at = int(0.75 * len(X))
X_test = X[split_at:]
X_train = X[:split_at]
y_test = y[split_at:]
y_train = y[:split_at]
print("=============================================================================")


X_test = np.asarray(X_test)
X_train = np.asarray(X_train)
y_test = np.asarray(y_test)
y_train = np.asarray(y_train)

print(X_test.shape)
print(y_test.shape)



model = Sequential()
## add layers, setting the inpot and the output
model.add(Dense(10, input_shape=(X.shape[1],), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(2, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())



print('Neural Network Model Summary: ')
print(model.summary())
# # Adam optimizer with learning rate of 0.001
# optimizer = Adam(lr=0.001)
# # selecet the suitable loss func, hint: one hot coding
# model.compile(optimizer, loss='', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, verbose=0, batch_size=5, epochs=2)


print("====================================================================================")
results = model.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
