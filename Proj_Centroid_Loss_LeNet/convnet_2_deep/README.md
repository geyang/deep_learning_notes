# convnet_2_deep README

@episodeyang 2016.

This is a package of a toy implementation of LeNet++ with 10 deep 
features instead of 2. 

The network topology look like this:

![network with 10 deep features](figures/Screenshot%202016-09-16%2011.34.23.png)

## Todos
- [ ] try to get the network to converge during training.

## Usage

### To Train

run
```shell
python3 MNIST_train.py
```


### To Evaluate

run
```shell
python3 MNIST_eval.py
```


### Prerequisite

Have Tensorflow installed, with GPU support. 

**NOTE** In order for the eval script to run at the same time as the 
training, you would need to have multiple devices (GPUs), or share the
same CPU (will be slow).
