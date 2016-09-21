# convnet_3_layer README

@episodeyang 2016.

This is a package of a toy implementation of LeNet++ with 50 deep 
features instead of 2, and three simple convolution layers.  

During training, the accuracy should rise up to 96% within 400 steps.

## Todos

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
