import numpy as np
from NeuralNetwork import *
from pylab import *
import pickle


def train_test_split(X, y, test_ratio):
        tr = int(y.size*test_ratio)
        return X[:tr], X[tr:], y[:tr], y[tr:]

def load_data(path):
        data = np.genfromtxt(path, delimiter=',', dtype=float)
        return data[:,:-1], data[:,-1].astype(int)

X, y = load_data("breastCancer/breast_cancer_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.75)
X_train, X_test = X_train.T, X_test.T
y_train = y_train[:,np.newaxis].T
y_test = y_test[:,np.newaxis].T
print(np.shape(y_test))
NN = NeuralNetwork([X_train.shape[0], 10, 1])
with open('bc_weights_bias/weights.pickle','rb') as handle:
	NN.weights = pickle.load(handle)
with open('bc_weights_bias/bias.pickle','rb') as handle:
	NN.bias = pickle.load(handle)
# with open('bc_weights_bias/weights.pickle','wb') as handle:
# 		pickle.dump(NN.weights,handle,protocol=pickle.HIGHEST_PROTOCOL)
# with open('bc_weights_bias/bias.pickle','wb') as handle:
# 		pickle.dump(NN.bias,handle,protocol=pickle.HIGHEST_PROTOCOL)

train_cost, train_accuracy = NN.SGD(X_train,y_train,0.0001,100,1)
test_accuracy = NN.accuracy(X_test,y_test)
print(test_accuracy)
figure()
plot(range(0,100),train_cost)
show()
