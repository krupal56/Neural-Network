##Nueral network


import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

class Neuron:
	def __init__(self,weights,bias):
		self.weights = weights
		self.bias = bias
	

	def feedforward(self,inputs):
		total = np.dot(self.weights,inputs) + self.bias
		return sigmoid(total)


weight = np.array([0,1])
bias = 4
n = Neuron(weight,bias)

x = np.array([2,3])
print (n.feedforward(x))


###

class OurNeuralNetwork:
  
  
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))

### 

def mse_loss(y_true,y_pred):
	return ((y_true-y_pred)**2).mean()

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)
		
		
