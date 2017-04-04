# ## Todo:
# - [ ] execute test after training
# 
# ## Done:
# - [x] a generative demo
# - [x] add validation (once an hour or so)
# - [x] add accuracy metric, use precision/recall.
# - [x] change to bi-directional GRU
# - [x] get data
# - [x] Add temperature to generator
# - [x] add self-feeding generator
# - [x] get training to work
# - [x] use optim and Adam
# 
# ## References
# 1: https://www.aclweb.org/anthology/D/D16/D16-1111.pdf  
# 2: https://phon.ioc.ee/dokuwiki/lib/exe/fetch.php?media=people:tanel:interspeech2015-paper-punct.pdf  
# 3: https://en.wikipedia.org/wiki/precision_and_recall  
# 4: https://en.wikipedia.org/wiki/F1_score  

# In[1]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import forward_tracer, backward_tracer, Char2Vec, num_flat_features

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np

from tqdm import tqdm

from IPython.display import clear_output, HTML

import os

from bs4 import BeautifulSoup




class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, bi=False):
        super(GruRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.bi_mul = 2 if bi else 1
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, self.layers, bidirectional=bi)
        self.decoder = nn.Linear(hidden_size * self.bi_mul, output_size)
        self.softmax = F.softmax
        
    def forward(self, x, hidden):
        #embeded = self.encoder(x)
        embeded = x
        #print(embeded.view(-1, 1, self.input_size).size())
        #print(hidden.size())
        gru_output, hidden = self.gru(embeded.view(-1, 1, self.input_size), hidden.view(self.layers * self.bi_mul, -1, self.hidden_size))
        #print(gru_output.size())
        output = self.decoder(gru_output.view(-1, self.hidden_size * self.bi_mul))
        return output, hidden
    
    def init_hidden(self, random=False):
        if random:
            return Variable(torch.randn(self.layers * self.bi_mul, self.hidden_size))
        else:
            return Variable(torch.zeros(self.layers * self.bi_mul, self.hidden_size)) 
"""
input_size = 105
hidden_size = 105
output_size = 105
layers = 2

gRNN = GruRNN(input_size, hidden_size, output_size, layers)

gRNN(Variable(torch.FloatTensor(10000, 105)),
     Variable(torch.FloatTensor(layers, 105)))"""


# In[12]:

class Engadget():
    def __init__(self, model, char2vec=None, output_char2vec=None):
        self.model = model
        if char2vec is None:
            self.char2vec = Char2Vec()
        else:
            self.char2vec = char2vec
            
        if output_char2vec is None:
            self.output_char2vec = self.char2vec
        else:
            self.output_char2vec = output_char2vec
            
        self.loss = 0
        self.losses = []
    
    def init_hidden_(self, random=False):
        self.hidden = model.init_hidden(random)
        return self
    
    def save(self, fn="GRU_Engadget.tar"):
        torch.save({
            "hidden": self.hidden, 
            "state_dict": model.state_dict(),
            "losses": self.losses
                   }, fn)
    
    def load(self, fn):
        checkpoint = torch.load(fn)
        self.hidden = checkpoint['hidden']
        model.load_state_dict(checkpoint['state_dict'])
        self.losses = checkpoint['losses']
    
    def setup_training(self, learning_rate):
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()
        
    def reset_loss(self):
        self.loss = 0
        
    def forward(self, input_text, target_text):
        
        self.hidden = self.hidden.detach()
        
        self.optimizer.zero_grad()
        self.next_(input_text)
        target_vec = Variable(self.output_char2vec.char_code(target_text))
        self.loss += self.loss_fn(self.output, target_vec)
        
    def descent(self):
        if self.loss is 0:
            print(self.loss)
            print('Warning: loss is zero.')
            return
        
        self.loss.backward()
        self.optimizer.step()
        self.losses.append(self.loss.cpu().data.numpy())
        self.reset_loss()
    
    def embed(self, input_data):
        self.embeded = Variable(self.char2vec.one_hot(input_data))
        return self.embeded
        
    def next_(self, input_text):
        self.output, self.hidden = self.model(self.embed(input_text), self.hidden)
        return self
    
    def softmax_(self, temperature=0.5):
        self.softmax = self.model.softmax(self.output/temperature)
        return self
    
    def output_chars(self, start=None, end=None):
        indeces = torch.multinomial(self.softmax[start:end]).view(-1)
        return self.output_char2vec.vec2list(indeces)

