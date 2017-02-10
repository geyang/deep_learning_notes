import numpy as np

DT = np.float32
eps = 1e-12
# Globals
components = []
params = []


# Global forward/backward
def Forward():
    for c in components:
        c.forward()


def Backward(loss):
    for c in components:
        if c.grad is not None: c.grad = DT(0)
    loss.grad = np.ones_like(loss.value)
    for c in components[::-1]:
        c.backward()


# Optimization functions
def SGD(lr):
    for p in params:
        lrp = p.opts['lr'] * lr if 'lr' in p.opts.keys() else lr
        p.value = p.value - lrp * p.grad
        p.grad = DT(0)


# Values
class Value:
    def __init__(self, value=None):
        self.value = DT(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DT(value).copy()


# Parameters
class Param:
    def __init__(self, value, opts={}):
        self.value = DT(value).copy()
        self.opts = {}
        params.append(self)
        self.grad = DT(0)


# Xavier initializer
def xavier(shape):
    sq = np.sqrt(3.0 / np.prod(shape[:-1]))
    return np.random.uniform(-sq, sq, shape)


# Utility function for shape inference with broadcasting
def bcast(x, y):
    xs = np.array(x.shape)
    ys = np.array(y.shape)
    pad = len(xs) - len(ys)
    if pad > 0:
        ys = np.pad(ys, [[pad, 0]], 'constant')
    elif pad < 0:
        xs = np.pad(xs, [[-pad, 0]], 'constant')
    os = np.maximum(xs, ys)
    xred = tuple([idx for idx in np.where(xs < os)][0])
    yred = tuple([idx for idx in np.where(ys < os)][0])
    return xred, yred


################################################### Actual components #####################################################


class Add:  # Add with broadcasting
    """
      Class name: Add
      Class usage: add two matrices x, y with broadcasting supported by numpy "+" operation.
      Class function:
          forward: calculate x + y with possible broadcasting
          backward: calculate derivative w.r.t to x and y, when calculate the derivative w.r.t to y, we sum up all the axis over grad except the last dimension.
    """

    def __init__(self, x, y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value + self.y.value

    def backward(self):
        xred, yred = bcast(self.x.value, self.y.value)
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(
                np.sum(self.grad, axis=xred, keepdims=True),
                self.x.value.shape)

        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.reshape(
                np.sum(self.grad, axis=yred, keepdims=True),
                self.y.value.shape)


class Mul:  # Multiply with broadcasting
    """
    Class Name: Mul
    Class Usage: elementwise multiplication with two matrix 
    Class Functions:
        forward: compute the result x*y
        backward: compute the derivative w.r.t x and y
    """

    def __init__(self, x, y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value * self.y.value

    def backward(self):
        xred, yred = bcast(self.x.value, self.y.value)
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(
                np.sum(self.grad * self.y.value, axis=xred, keepdims=True),
                self.x.value.shape)

        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.reshape(
                np.sum(self.grad * self.x.value, axis=yred, keepdims=True),
                self.y.value.shape)


class VDot:  # Matrix multiply (fully-connected layer)
    """
    Class Name: VDot
    Class Usage: matrix multiplication where x, y are matrices
    y is expected to be a parameter and there is a convention that parameters come last. Typical usage is x is batch feature vector with shape (batch_size, f_dim), y a parameter with shape (f_dim, f_dim2).
    Class Functions:
         forward: compute the vector matrix multplication result
         backward: compute the derivative w.r.t x and y, where derivative of x and y are both matrices
    """

    def __init__(self, x, y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = np.matmul(self.x.value, self.y.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.matmul(self.y.value, self.grad.T).T
        if self.y.grad is not None:
            nabla = np.matmul(self.x.value.T.reshape(list(self.x.value.shape) + [1]),
                              self.grad.reshape([1] + list(self.grad.shape)))
            self.y.grad = self.y.grad + nabla


class Log:  # Elementwise Log
    """
    Class Name: Log
    Class Usage: compute the elementwise log(x) given x.
    Class Functions:
        forward: compute log(x)
        backward: compute the derivative w.r.t input vector x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.log(self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad / self.x.value


class Sigmoid:
    """
    Class Name: Sigmoid
    Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 1/(1 + exp(-x_{i}))
    Class Functions:
        forward: compute activation y_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = 1. / (1. + np.exp(-self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * self.value * (1. - self.value)


class Tanh:
    """
    Class Name: Tanh
    Class Usage: compute the elementwise Tanh activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = (exp(x_{i}) - exp(-x_{i}))/(exp(x_{i}) + exp(-x_{i}))
    Class Functions:
        forward: compute activation y_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        x_exp = np.exp(self.x.value)
        x_neg_exp = np.exp(-self.x.value)

        self.value = (x_exp - x_neg_exp) / (x_exp + x_neg_exp)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (1 - self.value * self.value)


class RELU:
    """
    Class Name: RELU
    Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = max(0, x_{i})
    Class Functions:
        forward: compute activation y_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.maximum(self.x.value, 0)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (self.value > 0)


class LeakyRELU:
    """
    Class Name: LeakyRELU
    Class Usage: compute the elementwise LeakyRELU activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 0.01*x_{i} for x_{i} < 0 and y_{i} = x_{i} for x_{i} > 0
    Class Functions:
        forward: compute activation y_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.maximum(self.x.value, 0.01 * self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * np.maximum(0.01, self.value > 0)


class Softplus:
    """
    Class Name: Softplus
    Class Usage: compute the elementwise Softplus activation.
    Class Functions:
        forward: compute activation y_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.log(1. + np.exp(self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * 1. / (1. + np.exp(-self.x.value))


class SoftMax:
    """
    Class Name: SoftMax
    Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements in each batch (row). Specificaly, input is matrix [x_{00}, x_{01}, ..., x_{0n}, ..., x_{b0}, x_{b1}, ..., x_{bn}], output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(x_{bi})/(exp(x_{b0}) + ... + exp(x_{bn}))
    Class Functions:
        forward: compute probability p_{bi} for all b, i.
        backward: compute the derivative w.r.t input matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        lmax = np.max(self.x.value, axis=-1, keepdims=True)
        ex = np.exp(self.x.value - lmax)
        self.value = ex / np.sum(ex, axis=-1, keepdims=True)

    def backward(self):
        if self.x.grad is None:
            return
        gvdot = np.matmul(self.grad[..., np.newaxis, :], self.value[..., np.newaxis]).squeeze(-1)
        self.x.grad = self.x.grad + self.value * (self.grad - gvdot)


class LogLoss:
    """
    Class Name: LogLoss
    Class Usage: compute the elementwise -log(x) given matrix x. this is the loss function we use in most case.
    Class Functions:
        forward: compute -log(x)
        backward: compute the derivative w.r.t input matrix x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = -np.log(np.maximum(eps, self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + (-1) * self.grad / np.maximum(eps, self.x.value)


class Mean:
    """
    Class Name: Mean
    Class Usage: compute the mean given a vector x.
    Class Functions:
        forward: compute (x_{0} + ... + x_{n})/n
        backward: compute the derivative w.r.t input vector x
    """

    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.mean(self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * np.ones_like(self.x.value) / self.x.value.shape[0]

class Sum:
    """
    Class Name: Sum
    Class Usage: compute the sum of a matrix.
    """
    def __init__(self, x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.sum(self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * np.ones_like(self.x.value)


class MeanwithMask:
    """
    Class Name: MeanwithMask
    Class Usage: compute the mean given a vector x with mask.
    Class Functions:
        forward: compute x = x*mask and then sum over nonzeros in x/#(nozeros in x)
        backward: compute the derivative w.r.t input vector matrix
    """

    def __init__(self, x, mask):
        components.append(self)
        self.x = x
        self.mask = mask
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.sum(self.x.value * self.mask.value) / np.sum(self.mask.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * np.ones_like(self.x.value) * self.mask.value / np.sum(
                self.mask.value)


class Aref:  # out = x[idx]
    """
    Class Name: Aref
    Class Usage: get some specific entry in a matrix. x is the matrix with shape (batch_size, N) and idx
                 is vector contains the entry index and x is differentiable.
    Class Functions:
        forward: compute x[b, idx(b)]
        backward: compute the derivative w.r.t input matrix x
    """

    def __init__(self, x, idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        xflat = self.x.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.x.grad is not None:
            grad = np.zeros_like(self.x.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.x.grad = self.x.grad + grad


class Accuracy:
    """
    Class Name: Accuracy
    Class Usage: check the predicted label is correct or not. x is the probability vector where each probability is for each class. idx is ground truth label.
    Class Functions:
        forward: find the label that has maximum probability and compare it with the ground truth label.
        backward: None
    """

    def __init__(self, x, idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None

    def forward(self):
        self.value = np.mean(np.argmax(self.x.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


class Reshape:
    """
      Class name: Reshape
      Class usage: Reshape the tensor x to specific shape.
      Class function:
          forward: Reshape the tensor x to specific shape
          backward: calculate derivative w.r.t to x, which is simply reshape the income gradient to x's original shape
    """

    def __init__(self, x, shape):
        components.append(self)
        self.x = x
        self.shape = shape
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.reshape(self.x.value, self.shape)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(self.grad, self.x.value.shape)


def Momentum(lr, mom):
    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
            p.grad_hist = DT(0)
    for p in params:
        p.grad_hist = mom * p.grad_hist + p.grad
        p.grad = p.grad_hist
    SGD(lr)


def AdaGrad(lr, ep=1e-8):
    if 'grad_G' not in params[0].__dict__.keys():
        for p in params:
            p.grad_G = DT(0)
    for p in params:
        p.grad_G = p.grad_G + p.grad * p.grad
        p.grad = p.grad / np.sqrt(p.grad_G + DT(ep))
    SGD(lr)


def RMSProp(lr, g=0.9, ep=1e-8):
    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
            p.grad_hist = DT(0)
    for p in params:
        p.grad_hist = g * p.grad_hist + (1 - g) * p.grad * p.grad
        p.grad = p.grad / np.sqrt(p.grad_hist + DT(ep))
    SGD(lr)


_a_b1t = DT(1.0)
_a_b2t = DT(1.0)


def Adam(alpha=0.001, b1=0.9, b2=0.999, ep=1e-8):
    global _a_b1t
    global _a_b2t

    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
            p.grad_hist = DT(0)
            p.grad_h2 = DT(0)

    b1 = DT(b1)
    b2 = DT(b2)
    ep = DT(ep)
    _a_b1t = _a_b1t * b1
    _a_b2t = _a_b2t * b2
    for p in params:
        p.grad_hist = b1 * p.grad_hist + (1. - b1) * p.grad
        p.grad_h2 = b2 * p.grad_h2 + (1. - b2) * p.grad * p.grad

        mhat = p.grad_hist / (1. - _a_b1t)
        vhat = p.grad_h2 / (1. - _a_b2t)

        p.grad = mhat / (np.sqrt(vhat) + ep)
    SGD(alpha)


# clip the gradient if the norm of gradient is larger than some threshold, this is crucial for RNN.
def GradClip(grad_clip):
    for p in params:
        l2 = np.sqrt(np.sum(p.grad * p.grad))
        if l2 >= grad_clip:
            p.grad *= grad_clip / l2


##################################################### Recurrent Components ##############################################


class Embed:
    """
      Class name: Embed
      Class usage: Embed layer.
      Class function:
          forward: given the embeeding matrix w2v and word idx, return its corresponding embedding vector.
          backward: calculate the derivative w.r.t to embedding matrix
    """

    def __init__(self, idx, w2v):
        components.append(self)
        self.idx = idx
        self.w2v = w2v
        self.grad = None if w2v.grad is None else DT(0)

    def forward(self):
        self.value = self.w2v.value[np.int32(self.idx.value), :]

    def backward(self):
        if self.w2v.grad is not None:
            self.w2v.grad = np.zeros(self.w2v.value.shape)
            self.w2v.grad[np.int32(self.idx.value), :] = self.w2v.grad[np.int32(self.idx.value), :] + self.grad


class ConCat:
    """
      Class name: ConCat
      Class usage: ConCat layer.
      Class function:
          forward: concat two matrix along with the axis 1.
          backward: calculate the derivative w.r.t to matrix a and y.
    """

    def __init__(self, x, y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):

        self.value = np.concatenate((self.x.value, self.y.value), axis=1)

    def backward(self):

        dim_x = self.x.value.shape[1]
        dim_y = self.y.value.shape[1]

        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad[:, 0:dim_x]
        if self.y.grad is not None:
            self.y.grad = self.y.grad + self.grad[:, dim_x:dim_x + dim_y]


class ArgMax:
    """
      Class name: ArgMax
      Class usage: ArgMax layer.
      Class function:
          forward: given x, calculate the index which has the maximum value
          backward: None
    """

    def __init__(self, x):
        components.append(self)
        self.x = x

    def forward(self):
        self.value = np.argmax(self.x.value)

    def backward(self):
        pass
