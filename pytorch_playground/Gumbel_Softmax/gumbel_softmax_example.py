import torch
from torch.autograd import Variable
import torch.nn as nn

"""
Implemented from this paper: Categorical Reparameterization With Gumbel-Softmax

1. continuous distribution, approximate categorical samples.
2. outperform single-sample GD, in both Bernoulli and categorical.
3. semi-supervised training without marginalization over latent variable.



"""
class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()


