"""
torch vision helper module:

`datasets` contains popular datasets

- MNIST
- 

`transforms` contains data transformation functions.

`transforms.ToTensor()` => convert number to torch tensors.
`transforms.Normalize(mu1, mu2)` => normalizes the batch by these centroids.


"""
import torch
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True
)

from termcolor import cprint, colored as c

for batch_idx, (data, target) in enumerate(train_loader):
    cprint(
        c('batch data has the shape of: ', 'grey') +
        c(str(data), 'green')
    )
    break
