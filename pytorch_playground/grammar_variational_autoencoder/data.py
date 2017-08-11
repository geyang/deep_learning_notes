import h5py


def grammar_loader():
    with h5py.File('data/eq2_grammar_dataset.h5', 'r') as h5f:
        return h5f['data'][:]


def str_loader():
    with h5py.File('data/eq2_str_dataset.h5', 'r') as h5f:
        return h5f['data'][:]


GRAMMAR_DATA = grammar_loader()
print(GRAMMAR_DATA.shape)
# => shape(batch_size: 100000, seq_length: 15, tokens: 12)

STR_DATA = str_loader()
print(STR_DATA.shape)
# => shape(batch_size: 100000, seq_length: 19, tokens: 12)
