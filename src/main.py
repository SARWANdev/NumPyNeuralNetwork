import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data/mnist_train.csv")

# Transpose so every image is on it's own column 784 pixels + 1 label * 60000 images
data = np.array(data).T 

m, n = data.shape

y = data[0].astype(int) #Labels are the expected outputs  1st row
x = data[1:n].astype(float) / 255.0

def ReLU(z):
    return np.maximum(0,z)

def derivative_ReLU(z):
    return z>0

def softmax(z):
    # stable softmax over columns (examples in columns)
    z_shift = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def initialise_parameters():

    HIDDEN_LAYER_NEURONS = 20

    w1 = np.random.randn(HIDDEN_LAYER_NEURONS,784) #fills array with -0.5 to 0.5 randomly
    b1 = np.random.randn(HIDDEN_LAYER_NEURONS,1)

    w2 = np.random.randn(10,HIDDEN_LAYER_NEURONS) #fills array with -0.5 to 0.5 randomly
    b2 = np.random.randn(10,1)

    return w1, b1, w2, b2

def forward_propagation(w1, b1, w2, b2, x):

    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

def one_hot(y):
    m = y.size
    num_classes = int(y.max()) + 1
    one_hot_y = np.zeros((num_classes, m))
    one_hot_y[y, np.arange(m)] = 1
    return one_hot_y

def backward_propagation(z1,a1,z2,a2,w2, x, y):
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y                       # (10, m)
    dw2 = (1.0 / m) * dz2.dot(a1.T)            # (10, hidden)
    db2 = (1.0 / m) * np.sum(dz2, axis=1, keepdims=True)   # (10,1)
    dz1 = w2.T.dot(dz2) * derivative_ReLU(z1)  # (hidden, m)
    dw1 = (1.0 / m) * dz1.dot(x.T)             # (hidden, 784)
    db1 = (1.0 / m) * np.sum(dz1, axis=1, keepdims=True)   # (hidden,1)
    return dw1, db1, dw2, db2

def update_parameters( w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha*dw1
    w2 = w2 - alpha*dw2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    return w1,b1,w2,b2

def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions, y):
    with open("result.txt", "a") as results:
        results.write(f"\npredictions={predictions} ")
        results.write(f"expected values={y}")
    return np.sum(predictions==y)/y.size
       
def gradient_descent(x,y,iterations, alpha):
    w1, b1, w2, b2 = initialise_parameters()
    for i in range(iterations):
        z1,a1,z2,a2 = forward_propagation(w1, b1, w2, b2,x)
        dw1,db1,dw2,db2 = backward_propagation(z1,a1,z2,a2,w2, x, y)
        w1,b1,w2,b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i%50 == 0:
            with open("result.txt", "a") as results:
                results.write(f"Iteration: {i} ")
                results.write(f"Accuracy: {get_accuracy(get_predictions(a2),y)}")
    return w1,b1,w2,b2


w1,b1,w2,b2 = gradient_descent(x,y,iterations=1000,alpha=0.6)
test_data = pd.read_csv("data/mnist_test.csv")
test_data = np.array(data).T 
y_test = data[0].astype(int) #Labels are the expected outputs  1st row
x_test = data[1:n].astype(float) / 255.0
z1,a1,z2,a2 = forward_propagation(w1, b1, w2, b2,x_test)
with open("result.txt", "a") as results:
    results.write(f"Accuracy on Test Data:  {get_accuracy(get_predictions(a2),y_test)}")

