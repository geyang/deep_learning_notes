import numpy as np
from termcolor import cprint, colored as c


def forward_tracer(self, input, output):
    cprint(c("--> " + self.__class__.__name__, 'red') + " ===forward==> ")
    # print('')
    # print('input: ', type(input))
    # print('input[0]: ', type(input[0]))
    # print('output: ', type(output))
    # print('')
    # print('input size:', input[0].size())
    # print('output size:', output.data.size())
    # print('output norm:', output.data.norm())


def backward_tracer(self, input, output):
    cprint(c("--> " + self.__class__.__name__, 'red') + " ===backward==> ")


CHARS = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567890.,;\\\n\r\t~!@#$%^&*()-=_+<>?:"\'{}[]|\\`~\u00a0'
CHAR_DICT = {ch: i for i, ch in enumerate(CHARS)}


class Char2Vec():
    def __init__(self, size=None, chars=None):
        if chars is None:
            self.chars = CHARS
        else:
            self.chars = chars
        self.char_dict = {ch: i for i, ch in enumerate(CHARS)}
        if size:
            self.size = size
        else:
            self.size = len(CHARS)

    def one_hot(self, char, wrapper=None):
        one_hot = np.zeros(self.size)
        one_hot[self.char_dict[char]] = 1
        if wrapper:
            return wrapper(one_hot)
        return one_hot

    def char_code(self, char):
        return self.char_dict[char];

    def vec2char(vec):
        np.argmax(vec)


if __name__ == "__main__":
    # test
    print(Char2Vec(65).one_hot("B"))
    encoded = list(map(Char2Vec(65).one_hot, "Mary has a little lamb."))
    print(encoded)
