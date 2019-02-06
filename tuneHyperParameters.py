from NeuralNetwork import *
from ExtractMNIST import *
from pylab import *


def main():
	tr_x,tr_y,tv_x,tv_y,ts_x,ts_y = mnist()
	hidden_layer_neurons=50
	epochs = 100
	batch_size = 50
	lr = 0.5
	neurons_per_layer = [784,hidden_layer_neurons,10]
	NN = NeuralNetwork(neurons_per_layer)
	training_cost,tv_cost,accuracy_train,accuracy_tv =	NN.SGD(tr_x,tr_y,lr,epochs,batch_size,tv_x,tv_y)
	figure()
	plot(training_cost)
	plot(tv_cost)
	figure()
	plot(accuracy_train)
	plot(accuracy_tv)
	show()


if __name__ == "__main__":
	main()