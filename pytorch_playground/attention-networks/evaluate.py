# Evaluate


def print_random_pairs(model):
    input, output = data_loader.get_pair()
    hidden = model.init_hidden()
    model(input, output, hidden)

