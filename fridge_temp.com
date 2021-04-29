from logging import error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import keras

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but 


df = pd.read_csv("fridge_data.csv")
df = df.drop(["time", "Battery"], axis=1)

df = df[["inTemp"]]
df_original = df.values
column_names = list(df)

df_train = df[df["inTemp"]<14]

df_train = df_train.values
column_names = list(df)

fig, axs = plt.subplots(2)
fig.suptitle('Fridge data')
axs[0].plot(df_original)
axs[0].legend(["Original data"], loc="lower right")
axs[1].plot(df_train)
axs[1].legend(["only regular data"], loc="lower right")
plt.show()

# original_data_std = standardize(df_original)
# regular_data_std = standardize(df_train)


# make a lookback, this will prepare the data for the lstm model (num of samples * lookback * num of features)
lookback = 5
dataset = np.asarray(make_lookback(df_train, lookback))
dataset_with_anoms = np.asarray(make_lookback(df_original, lookback))
print('dataset ', dataset.shape)
print('dataset anoms ', dataset_with_anoms.shape)

# split train test if needed
X = np.asarray(dataset)
X_anoms = np.asarray(dataset_with_anoms)

#======================================================
# # constcut the model and train it
lstm_autoencoder = construct_model(X)
try:
    epochs = 2
    batch = 128
    lr = 0.001
    train_model(lstm_autoencoder, X, epochs = epochs, batch = batch, lr=lr)
except:
    save_model(lstm_autoencoder,'model_temp_amomaly.h5')
    print('saved for exception')

save_model(lstm_autoencoder,'model_temp_amomaly.h5')
print('saved normally')
#====================================================

lstm_autoencoder = keras.models.load_model('model_temp_amomaly.h5')

threshold = 0.3
# get the predcition for the irregular dataset names X_anoms
X_predictions = lstm_autoencoder.predict(X_anoms)

# calculate the error for each timestep, the result will be a (num of samples * num of featues)
print(X_anoms.shape)
print(X_predictions.shape)

err = calc_error_groups(X_anoms, X_predictions)
print("error shape", np.asarray(err).shape)

# get the indices of predictions that have an error above threshold
indices = get_pred_anom_indices(err, threshold=threshold)

# create anomly groups, (called neighbourhoods) out of the predicted anomaly points
neighborhoods = create_neighborhoods(indices, 5)

# flatten the data
flattened_pred = flatten_the_data(X_predictions, lookback)
error_flattened = flatten_the_data(err, lookback)

err_single = [x[0] for x in error_flattened]


fig, axs = plt.subplots(3)
fig.suptitle('irregular dataset reconstructed')

axs[0].plot(df_original)
axs[0].legend(["input"], loc="lower right")

axs[1].plot(flattened_pred)
axs[1].legend(["reconstructed input"], loc="lower right")


axs[2].plot(err_single)
axs[2].plot([0, len(err_single)], [threshold, threshold])
axs[2].legend(["error", "", "", "threshold"], loc="lower right")

plt.show()
