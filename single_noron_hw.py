import numpy as np
from numpy.lib.npyio import BagObj


# input X vector
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
# output Y vector
Y = [0, 0, 0, 1]

# weights
W = [1.41, 1.41]
B = 0

def step_func(input,step=0.5):
    if input >= step:  # ==0.5!
        return 1
    else:
        return 0

def feed_forward(record, weights):
    sigma = np.matmul(np.array(record),np.array(weights)) + B  # mat mul not dot product!
    print("sigma", sigma)
    return step_func(sigma)

def calculate_loss(output, target):
    return output-target

def delta_rule(x, weight, lr, loss):
    weight = weight - (lr * loss * x)
    return weight


epoch = 0
loss = 9999

while epoch < 5 or loss > 0:
    print("Epoch {}".format(epoch))
    epoch+=1

    for i,record in enumerate(X):
        print("Record {}, Input {}".format(i + 1, record))
        output = feed_forward(record, W)
        print(output)
        loss = calculate_loss(output, Y[i])
        print("The target is {}, Output is: {}, the loss {}".format(Y[i], output, loss))

        for j,w in enumerate(W):
            W[j] = delta_rule(record[j], w, 0.5, loss)
        print("Weights: ", W)
        print("========================================================")
    
