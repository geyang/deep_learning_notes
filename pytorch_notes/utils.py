import torch
from termcolor import cprint, colored as c


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


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


CHARS = "\x00 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567890.,;:?\"'\n\r\t~!@#$%^&*()-=_+<>{}[]|\\`~\xa0"
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

    def one_hot(self, source):
        y = torch.LongTensor([[self.char_dict[char]] for char in source])

        y_onehot = torch.zeros(len(source), self.size)
        y_onehot.scatter_(1, y, 1)

        return y_onehot

    def char_code(self, source):
        return torch.LongTensor([self.char_dict[char] for char in source])

    def vec2str(self, vec):
        chars = [self.chars[ind] for ind in vec.cpu().data.numpy()]
        return ''.join(chars)


if __name__ == "__main__":
    # test
    print(Char2Vec(65).one_hot("B"))
    encoded = list(map(Char2Vec(65).one_hot, "Mary has a little lamb."))
    print(encoded)
