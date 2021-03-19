"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
"""
import os

from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 



iris_data = load_iris()  # load the iris dataset
print('Example data: ')
print(iris_data.data[:5])
print('Example labels: ')
print(iris_data.target[:5])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)  # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.40, shuffle=False)

# Build the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())


fig = plt.figure(figsize=plt.figaspect(0.5))

# Train the model
for j in range(21):
    # model.train_on_batch(train_x,train_y)
    fig.suptitle('Epoch {}'.format(j))
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=1)
    prediction = model.predict(test_x).argmax(axis=1)
    #============================================================

    if j % 5 == 0:
        test_labels = test_y.argmax(axis=1)
        test_predicted = prediction
        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        i = 0
        for x,y,z,_ in test_x:
            if test_labels[i] == 0:
                ax.scatter(x,y,z, color="blue")
            elif test_labels[i] == 1:
                ax.scatter(x,y,z, color="red")
            else:
                ax.scatter(x,y,z, color="yellow")
            i = i + 1

        ax.set_xlabel("Petal width")
        ax.set_ylabel("Sepal length")
        ax.set_zlabel("Petal length")
        ax.set_title("real classes")

        #-----------------------------------------------
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        i = 0
        for x,y,z,_ in test_x:
            if test_predicted[i] == 0:
                ax.scatter(x,y,z, color="blue")
            elif test_predicted[i] == 1:
                ax.scatter(x,y,z, color="red")
            else:
                ax.scatter(x,y,z, color="yellow")
            i = i + 1

        ax.set_xlabel("Petal width")
        ax.set_ylabel("Sepal length")
        ax.set_zlabel("Petal length")
        ax.set_title("predicted classes")


        plt.pause(0.01)

plt.show()

# Test on unseen data

results = model.evaluate(test_x, test_y)
print("==============================================")
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))


our_record  = [[5.2, 3.6, 1.5, 0.3]]
predicted_class = model.predict(our_record) 
print("our record", our_record)
print("predicted class ",predicted_class)
print(train_x[0], train_y[0])

print("==============================================")
print("the confusion matrix")

print(confusion_matrix(test_y.argmax(axis=1), model.predict(test_x).argmax(axis=1)))
