import numpy as np
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

from model import GrammarVariationalAutoEncoder, VAELoss

from visdom_helper.visdom_helper import Dashboard


class Session():
    def __init__(self, model, train_step_init=0, lr=1e-3, is_cuda=False):
        self.train_step = train_step_init
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = VAELoss()
        self.dashboard = Dashboard('Grammar-Variational-Autoencoder-experiment')

    def train(self, loader, epoch_number):
        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        for batch_idx, data in enumerate(loader):
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            data = Variable(data)
            # do not use CUDA atm
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_fn(data, mu, log_var, recon_batch)
            loss.backward()
            self.optimizer.step()
            self.train_step += 1

            loss_value = loss.data.numpy()
            batch_size = len(data)

            self.dashboard.append('training_loss', 'line',
                                  X=np.array([self.train_step]),
                                  Y=loss_value / batch_size)

            if batch_idx == 0:
                print('batch size', batch_size)
            if batch_idx % 40 == 0:
                print('training loss: {:.4f}'.format(loss_value[0] / batch_size))
        return losses

    def test(self, loader):
        # nn.Module method, sets the training flag to False
        self.model.eval()
        test_loss = 0
        for batch_idx, data in enumerate(loader):
            data = Variable(data, volatile=True)
            # do not use CUDA atm
            recon_batch, mu, log_var = self.model(data)
            test_loss += self.loss_fn(data, mu, log_var, recon_batch).data[0]

        test_loss /= len(test_loader.dataset)
        print('testset length', len(test_loader.dataset))
        print('====> Test set loss: {:.4f}'.format(test_loss))


EPOCHS = 20
BATCH_SIZE = 200
import h5py


def kfold_loader(k, s, e=None):
    if not e:
        e = k
    with h5py.File('data/eq2_grammar_dataset.h5', 'r') as h5f:
        result = np.concatenate([h5f['data'][i::k] for i in range(s, e)])
        return torch.FloatTensor(result)


train_loader = torch.utils.data \
    .DataLoader(kfold_loader(10, 1),
                batch_size=BATCH_SIZE, shuffle=False)
# todo: need to have separate training and validation set
test_loader = torch.utils \
    .data.DataLoader(kfold_loader(10, 0, 1),
                     batch_size=BATCH_SIZE, shuffle=False)

losses = []
vae = GrammarVariationalAutoEncoder()

sess = Session(vae, lr=2e-3)
for epoch in range(1, EPOCHS + 1):
    losses += sess.train(train_loader, epoch)
    print('epoch {} complete'.format(epoch))
    sess.test(test_loader)
