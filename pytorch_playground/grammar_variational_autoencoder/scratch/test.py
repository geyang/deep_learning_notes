import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

x_stub = Variable(torch.DoubleTensor(100, 15, 12).normal_(0, 1))
conv_1 = nn.Conv1d(15, 15, 3)
y = conv_1(x_stub)
print(y)
