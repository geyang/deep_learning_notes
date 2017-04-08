import yaml
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import data, utils
from language import get_language_pairs
from visdom_helper import visdom_helper
from model import VanillaSequenceToSequence

print('********<loading train module>********')


def LTZeroInt(s):
    if int(s) > 0:
        return int(s)
    else:
        raise argparse.ArgumentTypeError('{} need to be larger than 1 and an integer.'.format(s))


_pr = argparse.ArgumentParser(description='Sequence-To-Sequence Model in PyTorch')

_pr.add_argument('-d', '--debug', dest='DEBUG', default=True, type=bool, help='debug mode prints more info')
_pr.add_argument('--input-lang', dest='INPUT_LANG', default='eng', help='code name for the input language ')
_pr.add_argument('--output-lang', dest='OUTPUT_LANG', default='cmn', help='code name for the output language ')
_pr.add_argument('--max-data-len', dest='MAX_DATA_LEN', default=10, type=int,
                 help='maximum length for input output pairs (words)')
_pr.add_argument('--dash-id', dest='DASH_ID', type=str, default='seq-to-seq-experiment',
                 help='maximum length for input output pairs')
_pr.add_argument('--batch-size', dest='BATCH_SIZE', type=int, default=10, help='maximum length for input output pairs')
_pr.add_argument('--learning-rate', dest='LEARNING_RATE', type=float, default=1e-3,
                 help='maximum length for input output pairs')
_pr.add_argument('--n-epoch', dest='N_EPOCH', type=int, default=5, help='number of epochs to train')
_pr.add_argument('-e', '--eval-interval', dest='EVAL_INTERVAL', type=LTZeroInt, default=10,
                 help='evaluate model on validation set')
_pr.add_argument('--teacher-forcing-r', dest='TEACHER_FORCING_R', type=float, default=0.5,
                 help='Float for the teacher-forcing ratio')
_pr.add_argument('-s', '--save-interval', dest='SAVE_INTERVAL', type=LTZeroInt, default=100,
                 help='evaluate model on validation set')
_pr.add_argument('--n-layers', dest='N_LAYERS', type=int, default=1,
                 help='maximum length for input output pairs')
_pr.add_argument('--bi-directional', dest='BI_DIRECTIONAL', type=bool, default=False,
                 help='whether use bi-directional module for the model')


class Session():
    def __init__(self, args):
        self.name = 'seq-2-seq-translator'
        self.args = args
        self.meta = lambda: None
        self.meta.epoch = 0

        # create logging and dashboard
        self.ledger = utils.Ledger(debug=self.args.DEBUG)
        self.dash = visdom_helper.Dashboard(args.DASH_ID)

        if self.args.DEBUG:
            self.ledger.pp(vars(args))

        # load data
        source_data = list(data.read_language(self.args.INPUT_LANG, self.args.OUTPUT_LANG, data.normalize_strings))
        self.ledger.green('sentence pairs in file: {}'.format(len(source_data)))
        self.pairs = list(filter(data.trim_by_length(self.args.MAX_DATA_LEN), source_data))
        self.ledger.green('Number of sentence pairs after filtering: {}'.format(len(self.pairs)))

        self.input_lang, self.output_lang = \
            get_language_pairs(self.args.INPUT_LANG, args.OUTPUT_LANG, self.pairs)
        self.input_lang.summarize()
        self.output_lang.summarize()

        self.word_index_pairs = [[self.input_lang.sentence_to_indexes(p[0]),
                                  self.output_lang.sentence_to_indexes(p[1])]
                                 for p in self.pairs]

        # Now build the model
        self.model = VanillaSequenceToSequence(
            self.input_lang, self.output_lang, 200, n_layers=self.args.N_LAYERS,
            bidirectional=self.args.BI_DIRECTIONAL)
        self.ledger.log('Sequence to sequence model graph:\n', self.model)

    def train(self, debug=False):
        # setting up Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # train
        self.meta.losses = []
        # print(self.word_index_pairs)
        for i, (inputs, target) in enumerate(tqdm(data.get_batch(self.word_index_pairs, self.args.BATCH_SIZE))):
            hidden = self.model.init_hidden(len(inputs))
            optimizer.zero_grad()

            input_vec = Variable(torch.LongTensor(inputs), requires_grad=False)
            target_vec = Variable(torch.LongTensor(target), requires_grad=False)
            _, target_seq_length = target_vec.size()

            # set the training flag
            self.model.train()
            # NOTE: we don't need to output longer than the target, because there is no way to calculate the entropy. This is however a very crude way of doing it, and in the future a better loss metric would be needed. self.args.MAX_OUTPUT_LEN
            output, hidden, output_embeded = self.model(input_vec, hidden, target_vec, self.args.TEACHER_FORCING_R,
                                                        target_seq_length - 1)

            b_size, seq_len, n_words = output_embeded.size()
            # TODO: instead of clipping output_embedded, should pad target longer with <NULL>. (different from t_padded!).
            # NOTE: need to cut target_vec to same length as seq_len
            target_without_SOS = target_vec[:, 1:(seq_len + 1)].contiguous().view(-1)
            # self.ledger.debug(output_embeded.size(), target_without_SOS.size())
            loss = criterion(output_embeded.view(-1, n_words), target_without_SOS)

            # target_vec.t()[1:].t().view(-1))

            # for output_softmax, t in zip(output_softmaxes, target_padded):
            #     loss += criterion(output_softmax, t)

            self.meta.losses.append(loss.data.numpy()[0])
            if i % self.args.EVAL_INTERVAL == 0:
                self.dash.plot('loss', 'line', X=np.arange(0, len(self.meta.losses)), Y=np.array(self.meta.losses))
            if i % self.args.SAVE_INTERVAL == 0:
                # TODO: add data file
                self.model.save()

            loss.backward()
            optimizer.step()

            # execute only once under debug mode.
            if debug:
                return
        self.meta.epoch += 1

    def checkpoint_location(self):
        return self.checkpoint_folder + '/' + self.name + '_batch_' + str(self.meta.epoch)

    def __enter__(self):
        try:
            self.meta = utils.Struct(**self.model.load(self.checkpoint_location()))
            if not self.args or self.args.LOAD_PARAMES_FROM_CHECKPOINT:
                self.args = dict(self.args.items() +i)
        except Exception as e:
            self.ledger.debug("Exception: ", e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.save(self.checkpoint_location(),
                        {
                            "args": args.
                            "meta": {
                                'losses': self.meta.losses,
                                'epoch': self.meta.epoch
                            }
                        })
        pass

    def evaluate(self, input_sentence):

        normalized = data.normalize_strings(input_sentence)
        input_ind = [self.input_lang.sentence_to_indexes(normalized)]
        input_vec = Variable(torch.LongTensor(input_ind), requires_grad=False)
        hidden = self.model.init_hidden(1)
        self.model.eval()
        translated, _, _ = self.model.forward(input_vec, hidden, max_output_length=1000)
        first_in_batch = 0
        output_sentence = self.output_lang.indexes_to_sentence(translated[first_in_batch].data.numpy())

        return output_sentence


if __name__ == "__main__":
    args = _pr.parse_args()
    sess = Session(args)
    for i in range(args.N_EPOCH):
        sess.train()
