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
        """remember, hidden layer always has batch_size at index = 1, reguardless of the batch_first flag."""
        if random:
            return Variable(
                torch.randn(
                    self.bi_multiplier * self.n_layers,
                    batch_size,
                    self.hidden_size))
        else:
            return Variable(
                torch.zeros(
                    self.bi_multiplier * self.n_layers,
                    batch_size,
                    self.hidden_size))


class DecoderRNN(nn.Module):
    def __init__(self, n_words, embedding_size, n_layers=1, bidirectional=False):
        """Need to impedance match input and output. Input is class_index, output is class_index, 
        but we also need the softmax raw during training, since it contains more information."""
        super(DecoderRNN, self).__init__()
        self.n_words = n_words
        self.hidden_size = embedding_size
        self.n_layers = n_layers
        if bidirectional:
            self.bi_multiplier = 2
        else:
            self.bi_multiplier = 1

        self.embedding = nn.Embedding(self.n_words, embedding_size)
        # add dropout
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.output_embedding = nn.Linear(self.hidden_size, n_words)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        """batch index goes first. Input and output are both size <,,n_words>"""
        batch_size = input.size()[0]
        embeded = self.embedding(input).view(batch_size, -1, self.hidden_size)
        output, hidden = self.gru(embeded, hidden)
        output_embeded = self.output_embedding(output.view(-1, self.hidden_size)).view(batch_size, -1, self.n_words)
        output_softmax = self.softmax(output_embeded.view(-1, self.n_words))
        _, output_words = output_softmax.topk(1, dim=1)#.view(batch_size, -1)
        # output_words = output_softmax.multinomial(1).view(batch_size, -1)
        return output_words, \
               hidden, \
               output_softmax


def init_hidden(self, batch_size, random=False):
    """remember, hidden layer always has batch_size at index = 1, reguardless of the batch_first flag."""
    if random:
        return Variable(
            torch.randn(
                self.bi_multiplier * self.n_layers,
                batch_size,
                self.hidden_size))
    else:
        return Variable(
            torch.zeros(
                self.bi_multiplier * self.n_layers,
                batch_size,
                self.hidden_size))  # TODO: attention


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
        batch_size = input.size()[0]
        encoder_output, encoded = self.encoder(input, hidden)
        start_input = Variable(torch.LongTensor([[self.output_lang.SOS_ind]] * batch_size))
        result = self.decoder(start_input, encoded)
        output, hidden, output_softmax = result
        output_length = output.size()[1]
        if output_length == 1:
            last_output = output
        else:
            last_output = output.index_select(1, Variable(torch.LongTensor([output_length - 1])))
        outputs = [last_output]
        output_softmaxes = [output_softmax]
        seq_length = target.size()[1]
        for i in range(seq_length):
            target_slice = torch.index_select(target, 1, Variable(torch.LongTensor([i]), requires_grad=False))
            # Advanced Indexing look here: https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497/2
            last_output, hidden, output_softmax = self.decoder(outputs[-1]
                                                               if random.random() > teacher_r else
                                                               target_slice, encoded)
            outputs.append(last_output)
            output_softmaxes.append(output_softmax)
        return outputs, output_softmaxes

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

    def load(self, fn):
        checkpoint = torch.load(fn)
        self.load_state_dict(checkpoint['state_dict'])
        # self.losses = checkpoint['losses']

    def save(self, fn="SeqToSeq.tar"):
        torch.save({
            "state_dict": self.state_dict(),
            # "losses": self.losses
        }, fn)
