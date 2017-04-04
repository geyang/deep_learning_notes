
# coding: utf-8

# # Learning Auto-Punctuation by Reading Engadget Articles
# 
# ## Overview
# 
# This project trains a bi-directional GRU to learn how to automatically punctuate a sentence by reading it character by character. The set of operation it learns include:
# ```
# capitalization: <cap>
#          comma:  ,
#         period:  .
#    dollar sign:  $
#      semicolon:  ;
#          colon:  :
#   single quote:  '
#   double quote:  "
#   no operation: <nop>
# ```
# 
# 
# ### Requirements
# 
# ```
# pytorch numpy matplotlib tqdm bs4
# ```
# 
# ## Model Setup and Considerations
# 
# The initial setup I began with was a single uni-direction GRU, with input domain [A-z0-9] and output domain of the ops listed above. My hope at that time was to simply train the RNN to learn correcponding operations. A few things jumped out during the experiment:
# 
# 1. **Use bi-directional GRU.** with the uni-direction GRU, the network quickly learned capitalization of terms, but it had difficulties with single quote. In words like "I'm", "won't", there are simply too much ambiguity from reading only the forward part of the word. The network didn't have enough information to properly infer such punctuations.
#     
#     So I decided to change the uni-direction GRU to bi-direction GRU. The result is much better prediction for single quotes in concatenations.
# 
#     the network is still training, but the precision and recall of single quote is nowt close to 80%.
#     
#     This use of bi-directional GRU is standard in NLP processes. But it is nice to experience first-hand the difference in performance and training.
#     
#     A side effect of this switch is that the network now runs almost 2x slower. This leads to the next item in this list:
# 2. **Use the smallest model possible.** At the very begining, my input embeding was borrowed from the Shakespeare model, so the input space include both capital alphabet as well as lower-case ones. What I didn't realize was that I didn't need the capital cases because all inputs were lower-case. 
#     
#     So when the training became painfully slow after I switch to bi-directional GRU, I looked for ways to make the training faster. A look at the input embeding made it obvious that half of the embedding space wasn't needed. 
# 
#     Removing the lower case bases made the traing around 3x faster. This is a rough estimate since I also decided to redownload the data set at the same time on the same machine.
#     
# 3. **Text formatting**. Proper formating of input text crawed from Engadget.com was crucial, especially because the occurrence of a lot of the puncuation was low and this is a character-level model. You can take a look at the crawed text inside [./engadget_data_tar.gz](./engadget_data_tar.gz). 
# 
# 4. **Async and Multi-process crawing is much much faster**. I initially wrote the engadget crawer as a single threaded class. Because the python `requests` library is synchronous, the crawler spent virtually all time waiting for the `GET` requests.
#     
#     This could be made a *lot* faster by parallelizing the crawling, or use proper async pattern. 
# 
#     This thought came to me pretty late during the second crawl so I did not implement it. But for future work, parallel and async crawler is going to be on the todo list.
# 
# 5. **Using Precision/Recall in a multi-class scenario**. The setup makes the reasonable assumption that each operation can only be applied mutually exclusively. The accuracy metric used here are **precision/recall** and the **F-score**, both commonly used in the literature<sup>1,</sup> <sup>2</sup>. The P/R and F-score are implemented according to wikipedia <sup>3,</sup> <sup>4</sup>.
#     
#     example accuracy output:
#     ```
#     Key: <nop>	Prec:  98.9%	Recall:  97.8%	F-Score:  98.4%
#     Key:   ,	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
#     Key: <cap>	Prec: 100.0%	Recall:  60.0%	F-Score:  75.0%
#     Key:   .	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
#     Key:   '	Prec:  66.7%	Recall: 100.0%	F-Score:  80.0%
#     ```
#     
# 6. **Hidden Layer initialization**: In the past I've found it was easier for the neural network to generate good results when both the training and the generation starts with a zero initial state. In this case because we are computing time limited, I zero the hidden layer at the begining of each file. 
# 
# ## Data and Cross-Validation
# 
# The entire dataset is composed of around 50k blog posts from engadget. I randomly selected 49k of these as my training set, 50 as my validation set, and around 0.5k as my test set. The training is a bit slow on an Intel i7 desktop, averaging 1.5s/file depending on the length of the file. As a result, it takes about a day to go through the entire training set.
# 
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


# In[17]:

layers = 1

model = GruRNN(input_size, hidden_size, output_size, layers=layers, bi=True)
egdt = Engadget(model, char2vec, output_char2vec)
#egdt.load('./data/Gru_Engadget_1_layer_bi_narrow.tar')


# In[18]:

learning_rate = 2e-3
egdt.setup_training(learning_rate)


# ## Training
# 
# The training below initializes the hidden layer at the beginning of each file. I believe this makes it more consistent, and easier for the network to converge.

# In[ ]:

model.zero_grad()
egdt.reset_loss()

seq_length = 1000

for epoch_num in range(10):
    
    step = 0
    
    for file_ind, (file_name, source) in enumerate(tqdm(train_gen())):
        
        # at the begining of the file, reset hidden to zero
        egdt.init_hidden_(random=False)
            
        for source_ in batch_gen(seq_length, source):
            
            
            step += 1
            
            input_source, punctuation_target = extract_punc(source_, egdt.char2vec.chars, egdt.output_char2vec.chars)
            #print(len(input_source), len(punctuation_target))
            
            try:
                egdt.forward(input_source, punctuation_target)
                if step%1 == 0:
                    egdt.descent()
                    
            except KeyError:
                print(source)
                raise KeyError
            

            if step%500 == 499:
                clear_output(wait=True)
                print('Epoch {:d}'.format(epoch_num))

                egdt.softmax_()

                plot_progress(egdt.embeded[:130].data.numpy(), 
                              egdt.output[:20].data.numpy(), 
                              egdt.softmax[:20].data.numpy(),
                              egdt.losses)
                
                punctuation_output = egdt.output_chars()
                result = apply_punc(input_source, punctuation_output)
                
                # print(punctuation_output, punctuation_target)
                print(result + "\n")
                
                print_pc(punctuation_output, punctuation_target)
                
        # validation, ran once in a while. takes a munite to run.
        if file_ind%200 == 1999:
            print('Dev Set Performance {:d}'.format(epoch_num))
