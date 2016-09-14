# convnet README

@episodeyang 2016.

This is a package of a simple ConvNet that learns the MNIST digit
dataset. 

## Todos
- [ ] use `tensorflow` app and FLAGS for the static parameters, to
reduce boilerplate code.

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
