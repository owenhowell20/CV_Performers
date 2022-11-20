#test
import numpy as np
import matplotlib.pyplot as plt

### use latex in plots
plt.rcParams.update({
	'font.size':16,
	'text.usetex': True,
	'text.latex.preamble':r'\usepackage{amsfonts}'
	})




### ambient dimension, how does kernel trick depend on ambient dimension?
### i.e. how many samples are needed to recreate the true softmax
d = 10

### number of samples in computing softmax
M = 10000


### compute the softmax inner product of x and y
def softmax(x , y):

	return np.exp( x.dot(y) )


### compute the aproxomate kernel using M samples
def compute_kernel(x,y,M):

	### sample M d random normal unit vectors
	omega = np.random.normal(0,1,size=(M,d) )

	### check how well the kernel aprox works
	avg = 0
	x_mag = x.dot(x)
	y_mag = y.dot(y)

	for m in range(M):

		x_dot =	omega[m,:].dot(x)
		y_dot = omega[m,:].dot(y)

		avg  = avg + (1/M)*np.exp( x_dot - 0.5*x_mag  ) * np.exp( y_dot - 0.5*y_mag  ) 


	return avg


### number of points to test on
N_points = 10000


### draw a x and y vector from gaussian distribution vectors with 1/sqrt(d) normilization
x_vects = np.random.normal(0,1,size=(N_points,d))/np.sqrt(d)
y_vects = np.random.normal(0,1,size=(N_points,d))/np.sqrt(d)


MSE = np.zeros(N_points)

for n in range(N_points):

	x = x_vects[n,:]
	y = y_vects[n,:]


	ker = compute_kernel(x,y,M)
	exact = softmax(x,y)

	MSE[n] = (ker - exact)**2



### plot histogram of the MSE
plt.grid()
plt.ylabel('Frequency')
plt.xlabel('$ (SM(X,Y) - \mathbb{E}[\hat{SM}(X,Y)] )^{2} $')
#plt.xscale('log')
plt.yscale('log')
plt.hist(MSE)
plt.tight_layout()
plt.show()


