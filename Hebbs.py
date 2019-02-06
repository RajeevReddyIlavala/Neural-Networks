import numpy as np
import itertools


def activationFunction(y_in):
	t_out=y_in.copy()
	t_out_abs=np.absolute(t_out[t_out!=0])
	t_out[t_out!=0]=t_out[t_out!=0]/t_out_abs
	return t_out

def isClassifiedCorrectly(t_out,t):
	if(np.array_equal(t_out,t)):
		return True
	else:
		return False

def flip(x,k,prod):
	combinations=[]
	positions=np.arange(0,15,1)
	result = list(itertools.combinations(positions,k))
	patternPool = np.full([len(result),x.shape[0]],x)
	n=0
	for combination in result:
		patternPool[n,[combination]]*=prod
		n+=1
	return patternPool


#input vectors
x=np.array([[-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1],[1,-1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1]])

#target outputs
t=np.array([1,-1])

#weights
w=np.full(15,0)
bias = 0

for i in range(0,x.shape[0]):
	w=w+t[i]*x[i]
	bias=bias+t[i]

y_in=x@np.transpose(w) + bias
t_out=activationFunction(y_in)

print("weights:",w)
print("bias:",bias)

if(isClassifiedCorrectly(t_out,t)):
	print("With the above weights and bias all the input patterns are classified correctly")
else:
	print("With the above weights and bias all the input patterns are not classified correctly")
print()


misclassified_corrupted = np.zeros((1,15), dtype =int)
misclassified_undetermined = np.zeros((1,15),dtype = int)
misclassified_patterns_corrupted=np.array([])
misclassified_patterns_undetermined=np.array([])
for k in range(1,16):
	patternPool_corrupted = flip(x[0,:],k,-1)
	patternPool_undetermined = flip(x[0,:],k,0)
	t_corrupted = activationFunction(patternPool_corrupted@np.transpose(w) + bias)
	t_undetermined = activationFunction(patternPool_undetermined@np.transpose(w)+bias)
	misclassified_corrupted[0,k-1]=np.sum(t_corrupted!=1)
	misclassified_undetermined[0,k-1]=np.sum(t_undetermined!=1)
	if(not misclassified_patterns_corrupted.size >0):
		misclassified_patterns_corrupted = patternPool_corrupted[t_corrupted!=1]
	if(not misclassified_patterns_undetermined.size>0):
		misclassified_patterns_undetermined = patternPool_undetermined[t_undetermined!=1]

print("No of patterns misclassified when 'k' pixels are flipped")
print("k=", np.arange(1,16,1))
print("#misclassifications:",misclassified_corrupted)
print("Sample misclassified pattern when 4 pixels are flipped:", misclassified_patterns_corrupted[0,:])
print()
print("No of patterns misclassified when 'k' pixels are undetermined")
print("#misclassifications:",misclassified_undetermined)
print("Sample misclassified pattern when 8 pixels are undetermined:", misclassified_patterns_undetermined[0,:])


