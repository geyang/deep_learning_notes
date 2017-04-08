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


import utils

CJK_LANGUAGES = ['cmn', 'jpn']  # , 'kor']


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
        self.index2word = {
            self.NULL_ind: self.NULL,
            self.SOS_ind: self.SOS,
            self.EOS_ind: self.EOS
        }
        self.n_words = 3
        self.CJK_LANGUAGES = CJK_LANGUAGES

    def add_sentence(self, sentence):
        for word in self.tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def tokenize(self, sentence):
        is_cjk = self.name in self.CJK_LANGUAGES
        if is_cjk:
            # ledger.warn('is CJK language!', lang.name)
            return [char for seg in sentence.split(' ') for char in seg]
        return sentence.split(' ')

    def sentence_to_indexes(self, sentence):
        return [self.word2index[w] for w in self.tokenize(sentence)]

    def indexes_to_sentence(self, indexes):
        ledger.debug(indexes[0])
        is_cjk = self.name in self.CJK_LANGUAGES
        if is_cjk:
            return ''.join([self.index2word[i] for i in indexes])
        return ' '.join([self.index2word[i] for i in indexes])

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
