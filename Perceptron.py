import numpy as np 

def transfer(y_in,theta):
	h=np.array([0,0])
	for i in range(0,2):
		if(y_in[i]>theta):
			h[i]=1
		elif(y_in[i]<-theta):
			h[i]=-1
		else:
			h[i]=0
	return h


x=np.array([[1, 1, 1,-1],[-1,1,1,-1],[1, 1, -1,-1],[-1,1,-1,-1],[1,-1, 1,1],[-1,-1,1,1],[1,-1,-1,1],[-1,-1,-1,1]])
x=np.transpose(x)
y=np.array([[-1,-1],[-1,-1],[-1,1],[-1,1],[1,-1],[1,-1],[1,1],[1,1]])
y=np.transpose(y)
w=np.zeros((2,5))

# x=np.array([[1, 1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]])
# x=np.transpose(x)
# y=np.array([[-1,-1],[-1,-1],[-1,1],[-1,1],[1,-1],[1,-1],[1,1],[1,1]])
# y=np.transpose(y)
# w=np.zeros((2,3))

x=np.vstack((np.ones(8),x))
print()
print("inputs::", x)
print()
print("y::", y)
print()
print("weights::",w)
print()
alpha=1
theta=0.2
count=0
i=0
classified = np.full(8,False)
while(True):
	count+=1
	print("iteration::",count)
	y_in = w@(x[:,i].reshape(x.shape[0],1))
	print("Y_in::",y_in)
	h=transfer(y_in,theta)
	print("h::",h)
	classified[i] = np.array_equal(h,y[:,i])
	if(not classified[i]):
		w=w+ (y[:,i].reshape(y.shape[0],1))@(x[:,i].reshape(1,x.shape[0]))
	print("Classified::",classified[i])
	print("New Weights::\n",w)
	if(np.all(classified)):
		break
	if(i==7):
		i=0
	else:
		i=i+1
	
print("No of iterations to confirm that the weights can be finalised::",count)



