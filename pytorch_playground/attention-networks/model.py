import torch
from torch.autograd import Variable
import torch.nn as nn

import utils


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
        self.output_softmax = nn.Softmax()

    def embed(self, input_word_indexes):
        """word index to embedding"""
        batch_size = input_word_indexes.size()[0]
        embeded = self.embedding(input_word_indexes).view(batch_size, -1, self.hidden_size)
        return embeded

    def forward(self, embeded, hidden):
        """batch index goes first. Input and output are both size <,,n_words>"""
        batch_size = embeded.size()[0]
        output, hidden = self.gru(embeded, hidden)
        output_embeded = self.output_embedding(output.view(-1, self.hidden_size)).view(batch_size, -1, self.n_words)
        return output_embeded, hidden

    def extract(self, output_embeded):
        """word embedding to class indexes"""
        b_size, seq_len, n_words = output_embeded.size()
        output_softmax = self.output_softmax(output_embeded.view(-1, n_words))
        # TODO: alternatively: output_words = output_softmax.multinomial(1).view(batch_size, -1)
        _, output_word_indexes = output_softmax.topk(1, dim=1)  # .view(batch_size, -1)
        return output_word_indexes.view(b_size, seq_len)


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

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)

    def get_SOS_vec(self, batch_size):
        return Variable(torch.LongTensor([[self.output_lang.SOS_ind]] * batch_size))

    def forward(self, input, hidden, target=None, teacher_r=0, max_output_length=None):
        # DONE: Should really use Module.train and Module.eval to set the training flag, then handle the different logic inside the forward function. This separate evaluation function is repetitive and will not needed in that case.
        # NOTE: hidden always has second index being the batch index.
        batch_size = hidden.size()[1]
        target_size = 0 if target is None else target.size()[1]
        encoder_output, encoded = self.encoder(input, hidden)
        slices = []
        embeded_outputs = []
        hidden = encoded
        output_word_batch = self.get_SOS_vec(batch_size)
        # TODO: make it so end of sentence for all elements in batch trigger end of while loop.
        eos_flags = list(range(batch_size))
        while len(slices) < (max_output_length or self.args.MAX_OUTPUT_LEN) \
                and len(eos_flags) != 0:
            # word_slice size(b_size, 1), is correct
            ## This is where you add teacher forcing
            index = len(slices)
            i_vec = Variable(torch.LongTensor([index]))
            # TODO: use tensor combine/add_with_mask operator here instead.
            output_slice_forced = output_word_batch \
                if random.random() > teacher_r or index >= target_size \
                else target.index_select(1, i_vec)
            output_embedded, hidden = self.decoder(self.decoder.embed(output_slice_forced), hidden)
            # convert embedded to class_index here
            output_word_batch = self.decoder.extract(output_embedded)

            # Now add the slices to the output stack. word_slice(b_size, 1) -> size(b_size)
            slices.append(output_word_batch.view(batch_size))
            embeded_outputs.append(output_embedded.view(batch_size, -1))

            for ind, s in enumerate(output_word_batch):
                s_index = int(s.data.numpy()[0])
                if ind in eos_flags:
                    if s_index == self.output_lang.EOS_ind:
                        eos_flags.remove(ind)

        # TODO: fix mismatch output between evaluate and forward.
        return torch.stack(slices, dim=1), hidden, torch.stack(embeded_outputs, dim=1)

    def setup_training(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()

    def load(self, fn):
        checkpoint = torch.load(fn)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint

    def save(self, fn="seq-to-seq.cp", meta=None, **kwargs):
        d = {k: kwargs[k] for k in kwargs}
        d["state_dict"] = self.state_dict()
        if meta is not None:
            d['meta'] = meta
        torch.save(d, fn)
