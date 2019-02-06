import numpy as np
from convert import *
from activationFuncs import *
import pickle
from pylab import *



def feed_forward(x_in,weights,bias):
		activation =x_in
		for w,b in zip(weights,bias):
			z = np.transpose(w)@activation + b
			a = sigmoid(z)
			activation = a
		return activation

def classify(input_pattern):
	with open('NN_Handwritten_weights/weights.pickle','rb') as handle:
		weights = pickle.load(handle)
	with open('NN_Handwritten_weights/bias.pickle','rb') as handle:
		bias = pickle.load(handle)
	activations = feed_forward(input_pattern,weights,bias)
	print(activations)
	y_bool =(activations>=0.5).squeeze()
	print(y_bool)
	pos = np.where(y_bool==True)
	digit = pos[0][0]
	return digit

path = 'Images/image.png'
input_pattern = modifyImage(path)
input_pattern[input_pattern<=10/255] =0
input_pattern[input_pattern<100]=input_pattern[input_pattern<100]*2
print(input_pattern*255)
digit = classify(np.transpose(input_pattern))
print("The digit is:",digit)
im = input_pattern.reshape(28,28)
figure()
gray()
imshow(im)
show()
