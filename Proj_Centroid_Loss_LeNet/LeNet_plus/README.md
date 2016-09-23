# LeNet_plus README

@episodeyang 2016.

This is a tensorflow implementation of the LeNet++ without the centroid
loss from Yandong Wen's paper 
[link](http://ydwen.github.io/papers/WenECCV16.pdf).

## Highlights

![2 D Deep Features](figures/MNIST%20LeNet++%20with%202%20Deep%20Features%20(PReLU).png)

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
