import torch
from torch.autograd import Variable
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, n_words, embedding_size, n_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.char_set = n_words
        self.hidden_size = embedding_size
        self.n_layers = n_layers
        if bidirectional:
            self.bi_multiplier = 2
        else:
            self.bi_multiplier = 1

        self.embedding = nn.Embedding(n_words, embedding_size)
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)

    def forward(self, input, hidden):
        """batch index goes first."""
        batch_size = input.size()[0]
        embeded = self.embedding(input).view(batch_size, -1, self.hidden_size)
        output, hidden = self.gru(embeded, hidden)
        return output, hidden

    def init_hidden(self, batch_size, random=False):
        if random:
            return Variable(
                torch.randn(batch_size,
                            self.bi_multiplier * self.n_layers,
                            self.hidden_size))
        else:
            return Variable(
                torch.zeros(batch_size,
                           self.bi_multiplier * self.n_layers,
                           self.hidden_size))


class DecoderRNN(nn.Module):
    def __init__(self, n_words, embedding_size, n_layers=1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = embedding_size
        self.n_layers = n_layers
        if bidirectional:
            self.bi_multiplier = 2
        else:
            self.bi_multiplier = 1

        self.embedding = nn.Embedding(n_words, embedding_size)
        # add dropout
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.output_embedding = nn.Linear(self.hidden_size, n_words)

    def forward(self, input, hidden):
        """batch index goes first. Input and output are both size <,,n_words>"""
        batch_size = input.size()[0]
        embeded = self.embedding(input).view(batch_size, -1, self.hidden_size)
        output, hidden = self.gru(embeded, hidden)
        return self.output_embedding(output), hidden  # , attn_weights

    def init_hidden(self, batch_size, random=False):
        if random:
            return Variable(torch.randn(batch_size,
                                        self.bi_multiplier * self.n_layers,
                                        self.hidden_size))
        else:
            return Variable(torch.zeros(batch_size,
                                       self.bi_multiplier * self.n_layers,
                                       self.hidden_size))


# TODO: attention
# 1. [ ] get training hooked up
#   1. [ ]

import random


class VanillaSequenceToSequence(nn.Module):
    def __init__(self, input_lang, output_lang, hidden_size, n_layers=1, bidirectional=False):
        super(VanillaSequenceToSequence, self).__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, bidirectional)
        self.decoder = DecoderRNN(output_lang.n_words, hidden_size, n_layers, bidirectional)

    # parameter collection is done automatically
    # def parameters(self):
    #     return [*self.encoder.parameters(), *self.decoder.parameters()]

    def forward(self, input, hidden, target, teacher_r):
        """target: [[int,],]"""
        encoder_output, encoded = self.encoder(input, hidden)
        output, hidden = self.decoder(self.output_lang.SOS_ind, encoded)
        outputs = [output]
        # teacher forcing, terminates for target length.
        for i, target_w in enumerate(target):
            output, hidden = self.decoder(outputs[-1] if random.radom() > teacher_r else target_w, encoded)
            outputs.append(output)
        return outputs

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)

    def _evaluate(self, input, hidden, max_output_length=100):
        encoder_output, encoded = self.encoder(input, hidden)
        sentence = []
        i = 0
        hidden = encoded
        word = self.SOS
        while i <= max_output_length and word != self.output_lang.EOS_ind:
            word, hidden = self.decoder(word, hidden)
            sentence.append(word)
        return sentence

    def setup_training(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()
