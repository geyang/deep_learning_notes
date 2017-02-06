import numpy as np
from collections import Counter

# some global variable.
vocab = dict()
vocab['@'] = 0
n_vocab = 0

#####  char language modeling helper functions
# pad the seq with 0    
def make_mask(data):

    length = [len(seq) for seq in data]
    max_len = max(length)
    masked_data = []
    
    for i, seq in enumerate(data):
        seq_pad = np.zeros(max_len, dtype=np.int32)
        seq_pad[0:length[i]] = seq
        masked_data.append(seq_pad)
        
    return masked_data
    
# load the data and transform the char to integer.    
def load_data_onechar(filename):

    global vocab, n_vocab
    dataset = []
    word_cnt = 0
    lines = open(filename,"rb").readlines()

    for line in lines:
        raw = '{'+line.decode('utf-8', errors='replace').strip().lower()+'}'
        chars = [char for char in raw]
        word_cnt += len(chars)
        idx = np.ndarray((len(chars),), dtype=np.int32)
        for i, char in enumerate(chars):
            if char not in vocab:
                vocab[char] = len(vocab)
            idx[i] = vocab[char]
        dataset.append(idx)
            
    n_vocab = len(vocab) 
    dataset.sort(key=len)
    
    return dataset, word_cnt

# given integer and return the corresponding the char
def to_word(idx):
    global vocab
    return list(vocab.keys())[list(vocab.values()).index(idx)]

# given the corresponding the char and return the index 
def to_index(word):
    global vocab
    return vocab.get(word.lower(), vocab.get('{', 0))

# given integers and return the corresponding the chars
def to_string(idxs):
    string = [to_word(idx) for idx in idxs]
    return ''.join(string) 

# given the corresponding the chars and return the indexes. 
def to_idxs(words):
    idxs = [to_index(word) for word in words] 
    return idxs 