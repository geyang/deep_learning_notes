import numpy as np

DT=np.float32
eps=1e-12

# Globals
components = []
params = []
batch = 0

# Global forward/backward
def Forward():
    for c in components: c.forward()

def Backward(loss):
    for c in components:
        if c.grad is not None: c.grad = DT(0)
    loss.grad = np.ones_like(loss.value)
    for c in components[::-1]: c.backward()

# Optimization functions with SGD algorithm
def SGD(eta):
    for p in params:
        p.value = p.value - eta*p.grad
        p.grad = DT(0)

# Values
class Value:
    def __init__(self,value=None):
        self.value = DT(value).copy()
        self.grad = None

    def set(self,value):
        self.value = DT(value).copy()

# Parameters
class Param:
    def __init__(self,value,opts = {}):
        self.value = DT(value).copy()
        self.opts = {}
        params.append(self)
        self.grad = DT(0)

# Xavier initializer
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


####################################### Actual Component  #####################################
'''
  Class name: Add
  Class usage: add  matrix x with vector y with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate x + y with possible broadcasting
      backward: calculate derivative w.r.t to x and y
'''
class Add:
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value + self.y.value

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad

        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.mean(self.grad, axis=0)
 
'''
Class Name: VDot
Class Usage: matrix multiplication where x, y are matrices
y is expected to be a parameter and there is a convention that parameters come last. Typical usage is x is batch feature vector with shape (batch_size, f_dim), y a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t x and y, where derivative of x and y are both matrices 
'''           
class VDot:
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = np.dot(self.x.value,self.y.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.dot(self.grad, self.y.value.T)
        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.dot(self.x.value.T, self.grad)/self.x.value.shape[0] # take the mean gradient

            
'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 1/(1 + exp(-x_{i}))
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class Sigmoid:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = 1. / (1. + np.exp(-self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * self.value * (1.-self.value)


'''
Class Name: Tanh
Class Usage: compute the elementwise Tanh activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = (exp(x_{i}) - exp(-x_{i}))/(exp(x_{i}) + exp(-x_{i}))
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class Tanh:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):

        x_exp = np.exp(self.x.value)
        x_neg_exp = np.exp(-self.x.value)
        self.value = (x_exp - x_neg_exp)/(x_exp + x_neg_exp)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (1 - self.value*self.value)

'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = max(0, x_{i})
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class RELU:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.maximum(self.x.value, 0)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (self.value > 0)
            
'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements in each batch (row). Specificaly, input is matrix [x_{00}, x_{01}, ..., x_{0n}, ..., x_{b0}, x_{b1}, ..., x_{bn}], output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(x_{bi})/(exp(x_{b0}) + ... + exp(x_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix x 
'''            
class SoftMax:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        lmax = np.max(self.x.value,axis=-1,keepdims=True)
        ex = np.exp(self.x.value - lmax)
        self.value = ex / np.sum(ex,axis=-1,keepdims=True)

    def backward(self):
        if self.x.grad is None:
            return
        
        gvdot = np.matmul(self.grad[...,np.newaxis,:],self.value[...,np.newaxis]).squeeze(-1) # b*b, b*1*b
        self.x.grad = self.x.grad + self.value * (self.grad - gvdot)
        
'''
Class Name: LogLoss
Class Usage: compute the elementwise log(x) given vector x.
Class Functions:
    forward: compute log(x)
    backward: compute the derivative w.r.t input matrix x
'''        
class LogLoss:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = -np.log(np.maximum(eps,self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + (-1)*self.grad/np.maximum(eps,self.x.value) # x.grad = b*1
            
            global batch
            # obtain the batch size when backprop
            batch = self.x.grad.shape[0]
            
'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. x is the matrix with shape (batch_size, N) and idx is vector contains the entry index and x is differentiable.
Class Functions:
    forward: compute x[b, idx(b)]
    backward: compute the derivative w.r.t input matrix x
'''            
class Aref: # out = x[idx]
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        xflat = self.x.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat)/outer_dim
        self.pick = np.int32(np.array(range(outer_dim))*inner_dim+iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.x.grad is not None:
            grad = np.zeros_like(self.x.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.x.grad = self.x.grad + grad  # batch*label  

            
'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. x is the probability vector where each probability is for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''            
class Accuracy:
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None

    def forward(self):
        self.value = np.mean(np.argmax(self.x.value,axis=-1)==self.idx.value)

    def backward(self):
        pass