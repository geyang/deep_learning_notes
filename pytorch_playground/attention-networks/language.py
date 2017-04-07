# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
from utils import ledger


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


CJK_LANGUAGES = ['cmn', 'jpn']  # , 'kor']


def tokenize(sentence, is_cjk=False):
    if is_cjk:
        tokens = [char for seg in sentence.split(' ') for char in seg]
        return tokens
    return sentence.split(' ')


import utils


class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.NULL = "<NULL>"
        self.NULL_ind = 0
        self.SOS = '<SOS>'
        self.SOS_ind = 1
        self.EOS = '<EOS>'
        self.EOS_ind = 2
        self.index2word = {self.NULL_ind: self.NULL, self.SOS_ind: self.SOS, self.EOS_ind: self.EOS}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in tokenize(sentence, is_cjk=(self.name in CJK_LANGUAGES)):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def pad_target(self, target_unpadded):
        return [
            [self.SOS_ind] + s for s in target_unpadded
        ]

    def summarize(self):
        utils.ledger.green('{}'.format(self.name), end=' ')
        utils.ledger.print('has ', end=' ')
        utils.ledger.green(self.n_words, end=' ')
        utils.ledger.print('words', end='.\n')


def get_language_pairs(name_1, name_2, sentence_pairs):
    l1 = Language(name_1)
    l2 = Language(name_2)
    for p in sentence_pairs:
        l1.add_sentence(p[0])
        l2.add_sentence(p[1])
    return l1, l2


if __name__ == "__main__":
    l = Language('eng')
