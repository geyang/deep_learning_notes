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
_pr.add_argument('-s', '--save-interval', dest='SAVE_INTERVAL', type=LTZeroInt, default=100,
                 help='evaluate model on validation set')
_pr.add_argument('--n-layers', dest='N_LAYERS', type=int, default=1,
                 help='maximum length for input output pairs')
_pr.add_argument('--bi-directional', dest='BI_DIRECTIONAL', type=bool, default=False,
                 help='whether use bi-directional module for the model')


class Session():
    def __init__(self, args):
        self.args = args

        # create logging and dashboard
        self.ledger = utils.Ledger(debug=self.args.DEBUG)
        self.dash = visdom_helper.Dashboard(args.DASH_ID)

        if self.args.DEBUG:
            self.ledger.pp(vars(args))

        # load data
        source_data = data.read_language(self.args.INPUT_LANG, self.args.OUTPUT_LANG, data.normalize_strings)
        self.pairs = list(filter(data.trim_by_length(self.args.MAX_DATA_LEN), source_data))
        self.ledger.green('Number of sentence pairs after filtering: {}'.format(len(self.pairs)))

        self.input_lang, self.output_lang = \
            get_language_pairs(self.args.INPUT_LANG, args.OUTPUT_LANG, self.pairs)
        self.input_lang.summarize()
        self.output_lang.summarize()

        self.word_index_pairs = [[data.sentence_to_indexes(self.input_lang, p[0]),
                                  data.sentence_to_indexes(self.output_lang, p[1])]
                                 for p in self.pairs]

        # Now build the model
        self.model = VanillaSequenceToSequence(
            self.input_lang, self.output_lang, 200, n_layers=self.args.N_LAYERS,
            bidirectional=self.args.BI_DIRECTIONAL)
        self.ledger.log('Sequence to sequence model graph:\n', self.model)

    def train(self):
        # setting up Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.LEARNING_RATE)
        criterion = nn.NLLLoss()

        # train
        losses = []
        # print(self.word_index_pairs)
        for i, (inputs, target) in enumerate(tqdm(data.get_batch(self.word_index_pairs, self.args.BATCH_SIZE))):
            hidden = self.model.init_hidden(len(inputs))
            optimizer.zero_grad()

            input_vec = Variable(torch.LongTensor(inputs), requires_grad=False)
            target_vec = Variable(torch.LongTensor(target), requires_grad=False)

            outputs, output_softmaxes = self.model(
                input_vec,
                hidden,
                target_vec,
                0.5
            )
            loss = 0
            target_padded = Variable(torch.from_numpy(np.array(self.output_lang.pad_target(target)).T))
            for output_softmax, t in zip(output_softmaxes, target_padded):
                loss += criterion(output_softmax, t)

            losses.append(loss.data.numpy()[0])
            if i % self.args.EVAL_INTERVAL == 0:
                self.dash.plot('loss', 'line', X=np.arange(0, len(losses)), Y=np.array(losses))
            if i % self.args.SAVE_INTERVAL == 0:
                # TODO: add data file
                self.model.save()

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    args = _pr.parse_args()
    sess = Session(args)
    for i in range(args.N_EPOCH):
        sess.train()
