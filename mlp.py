import numpy as np

class Neuron:
    def __init__(self, weights=None):
        self.w = weights

    def activation(self, data):
        return int(np.heaviside(data, 1))
    
    def forward(self, data):
        a = np.dot(data, self.w)
        return self.activation(a)

# layer1
n1_1 = Neuron(weights = [-.5, 1, 1]) 
n1_2 = Neuron(weights = [-1.5, 1, 1])

# layer 2
n2_1 = Neuron(weights = [-.5, 1, -2])

def XOR(x1, x2):
    # pass through layer 1
    z1 = n1_1.forward([1, x1, x2])
    z2 = n1_2.forward([1, x1, x2])

    # pass thru layer 2
    output = n2_1.forward([1, z1, z2])
    return output

print ("XOR(0, 0)", XOR(0, 0))
print ("XOR(0, 1)", XOR(0, 1))
print ("XOR(1, 0)", XOR(1, 0))
print ("XOR(1, 1)", XOR(1, 1))