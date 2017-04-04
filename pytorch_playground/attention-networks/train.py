import data as data
from model import VanillaSequenceToSequence

source_data = list(data.read_language('eng', 'fra', data.normalize_strings))

data.log('Number of sentence pairs: {}'.format(len(source_data)))
MAX_LEN = 10

ps_filtered = list(filter(data.trim_by_length(MAX_LEN), source_data))
data.log('Number of sentence pairs after filtering: {}'.format(len(ps_filtered)))

from language import get_language_pairs

eng, fra = get_language_pairs('eng', 'fra', ps)

word_index_pairs = [data.sentence_to_indexes(eng, p[0]), data.sentence_to_indexes(fra, p[1])
for p in ps_filtered]

import torch
from torch.autograd import Variable

# In[]: Build the model
BATCH_SIZE = 10
seq2seq = VanillaSequenceToSequence(eng, fra, 200, n_layers=1, bidirectional=False)

# now train
for i, (inputs, outputs) in enumerate(data.get_batch(word_index_pairs, BATCH_SIZE)):
    print('--------')
    hidden = seq2seq.init_hidden(len(inputs))
    seq2seq(
        Variable(torch.Tensor(inputs)),
        hidden,
        Variable(torch.Tensor(outputs)),
        0.5
    )
    print(seq2seq.parameters())
