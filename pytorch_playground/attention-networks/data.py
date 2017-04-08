import re

import language
from utils import ledger


def normalize_strings(s):
    s = s.lower().strip()
    # http://stackoverflow.com/a/518232/2809427
    # disable below since it can be learned.
    # s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?;:@])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_language(l1, l2, normalize_fn=None):
    ledger.info('Reading Lines... ')
    with open('./training-data/{}-{}.txt'.format(l1, l2)) as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            if normalize_fn is None:
                yield l.split('\t')
            else:
                yield [normalize_fn(s) for s in l.split('\t')]


def trim_by_length(length, token_sep=' '):
    def trim(p):
        if length <= 0:
            return True
        elif len(p[0].split(token_sep)) > length or len(p[1].split(token_sep)) > length:
            return False
        return True

    return trim


import math


def get_batch(pairs, batch_size):
    n_pairs = len(pairs)
    n_batch = math.ceil(n_pairs / batch_size)
    for i in range(n_batch):
        sent_1 = []
        sent_2 = []
        # for batch_size 1 no padding is needed
        if batch_size == 1:
            p = pairs[i]
            sent_1.append(p[0])
            sent_2.append(p[1])
            yield sent_1, sent_2
        else:
            input_max_length = 0
            output_max_length = 0
            for p in pairs[i * batch_size:min((i + 1) * batch_size, n_pairs)]:
                input_max_length = max(len(p[0]), input_max_length)
                output_max_length = max(len(p[1]), output_max_length)

            for p in pairs[i * batch_size:min((i + 1) * batch_size, n_pairs)]:
                # [2] is [<EOS>], [0] [<NULL>]
                sent_1.append([1] + p[0] + [2] + [0] * (input_max_length - len(p[0])))
                sent_2.append([1] + p[1] + [2] + [0] * (output_max_length - len(p[1])))
            yield sent_1, sent_2


if __name__ == "__main__":
    ledger.green('Number of sentence pairs: {}'.format(len(list(read_language('eng', 'fra', normalize_strings)))))
    max_len = 10

    ps = filter(trim_by_length(max_len), read_language('eng', 'fra', normalize_strings))
    ps = list(ps)
    ledger.green('Number of sentence pairs after filtering: {}'.format(len(ps)))

    l1, l2 = language.get_language_pairs('eng', 'fra', ps)
