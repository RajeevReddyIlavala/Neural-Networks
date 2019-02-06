import numpy as np
from pylab import *
from math import *
from mpl_toolkits.mplot3d import Axes3D

class Adaline:
	def __init__(self,counts):
		self.no_of_layers = len(counts)
		self.neurons_per_layer = counts
		self.weights=[]
		self.bias = []
		for n in range(0,self.no_of_layers-1):
			self.weights.append(np.zeros((self.neurons_per_layer[n],self.neurons_per_layer[n+1])))
			self.bias.append(np.zeros((self.neurons_per_layer[n+1],1)))

	def feed_forward(self,x_in):
		activations = [x_in]
		for w,b in zip(self.weights,self.bias):
			z = np.transpose(w)@activations[-1] + b
			activations.append(z)
		return activations

	def backprop(self,a,y):
		sensitivities = []
		sensitivities.append(a - y)
		for i in range(self.no_of_layers-2,0,-1):
			s = self.weights[i]@sensitivities[0]
			sensitivities.insert(0,s)
		return sensitivities

	def costFunction(self,y,t):
		return np.sum(np.square(y-t))/t.shape[1]

	def accuracy(self,x_in,y):
		activations = self.feed_forward(x_in)
		t = activations[-1].copy()
		t[t>=0] = 1
		t[t<0] = -1
		count = 0
		for i in range(0,y.shape[1]):
			if(np.array_equal(t[:,i],y[:,i])):
				count+=1
		accuracy=(count/y.shape[1])* 100
		return (count,accuracy)



	def trainNN(self,x_in,y,alpha):
		J=np.empty((1,2))
		iter=1
		drop = 0.1
		step_size = 3
		while(iter<11000):
			# if(iter%11 ==0):
			# 	alpha =alpha*0.99
			# epoch  = 1+floor((iter)/11)
			# alpha = alpha * (drop**floor(epoch/step_size))
			for n in range(0,x_in.shape[1]):
				activations = self.feed_forward(x_in[:,n][:,np.newaxis])
				sensitivities = self.backprop(activations[-1],y[:,n][:,np.newaxis])
				for l in range(0,self.no_of_layers-1):
					self.weights[l] = self.weights[l] - 2*alpha * activations[l]@np.transpose(sensitivities[l])
					self.bias[l] = self.bias[l]-2*alpha*sensitivities[l]
				cost = self.costFunction(self.feed_forward(x_in)[-1],y)
				J=np.vstack((J,[iter,cost]))
				iter+=1
		return J




def main():
	s = np.array([[1,1,-1],[1,2,-1],[2,-1,1],[2,0,1],[1,-2,1],[0,0,1],
		[-1,2,1],[-2,1,1],[-1,-1,-1],[-2,-2,-1],[-2,-1,-1]])
	t = np.array([[-1,-1],[-1,-1],[-1,1],[-1,1],[-1,1],[-1,1],[1,-1],[1,-1],[1,1],[1,1],[1,1]])

	counts=[3,2]
	alpha = 0.00035
	NN = Adaline(counts)
	s=np.transpose(s)
	t=np.transpose(t)
	J=NN.trainNN(s,t,alpha)
	(count,accuracy) = NN.accuracy(s,t)
	y = (np.transpose(NN.weights[0]) @ s) + NN.bias[0]
	y_transfer = y.copy()
	y_transfer[y_transfer>=0] = 1
	y_transfer[y_transfer<0]=-1
	cost_transfer = np.sum(np.square(y_transfer-t))/t.shape[1]
	print("weights:",NN.weights[0])
	print("bias:",NN.bias[0])
	print("No of patterns classified correctly:",count)
	print("Training accuracy:",accuracy)
	print("Cost after training:",J[-1,1])
	print("Cost after applying transfer function:", cost_transfer)	
	figure()
	plot(J[:,0],J[:,1])
	figure()
	ax = axes(projection='3d')
	ax.scatter(s[0,:2],s[1,:2],s[2,:2])
	ax.scatter(s[0,2:6],s[1,2:6],s[2,2:6])	
	ax.scatter(s[0,6:8],s[1,6:8],s[2,6:8])
	ax.scatter(s[0,8:11],s[1,8:11],s[2,8:11])
	show()
	



if __name__ == "__main__":
		main()





