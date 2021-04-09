from functions import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

df = pd.read_csv("processed_dataset.csv")
feature_names = list(df)
print(feature_names)


X_lstm = df[["TKLF3", "TR1", "TR2","FgHeater","AmbTemp" ,"Energy"]].values


std_scaler = StandardScaler()
std_scaler.fit(X_lstm)
X_lstm = std_scaler.transform(X_lstm)


X = [x[:5] for x in X_lstm]
y = [x[5] for x in X_lstm]


X_lb = make_lookback(X, 10)
y_lb = make_lookback(y, 10)
# y_ = [x[-1] for x in y_lb]

y_ = [x[0] for x in y_lb]
X_lb = X_lb[:-1]  # discard the last record
y_ = y_[1:]  # discard first target record


X_lb = np.asarray(X_lb)
print(X_lb.shape)
y_ = np.asarray(y_)
print(y_.shape)


# split according to the length of the dataset, hint: x = x[:split_at]
split_at = int(0.75 * len(X_lb))
X_test = X_lb[split_at:]
X_train = X_lb[:split_at]
y_test = y_[split_at:]
y_train = y_[:split_at]


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test).reshape(-1,1)
y_train = np.asarray(y_train).reshape(-1,1)

print("=============================================================================")
lstm_model = construct_LSTM(X_train.shape, 1)
lstm_history = lstm_model.fit(X_train, y_train,
                              epochs=10,
                              batch_size=128,
                              validation_data=(X_test, y_test),
                              verbose=1
                              ).history


lstm_model.save("lstm_model.h5")
print("====================================================================================")

lstm_model = keras.models.load_model('lstm_model.h5')
results = lstm_model.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

y_pred = lstm_model.predict(X_test)

fig, axs = plt.subplots(3)
fig.suptitle('LSTM Results')

axs[0].plot(y_train)
axs[0].legend(["trained to predict"], loc ="lower right") 


axs[1].plot(y_test, color="black")
axs[1].legend(["test data"], loc ="lower right") 

axs[2].plot(y_pred,color="orange")
axs[2].legend(["pred_data data"], loc ="lower right") 

plt.show()
