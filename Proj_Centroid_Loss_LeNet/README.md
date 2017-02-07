# LeNet_plus_centerloss README

@episodeyang 2016.

This is a Tensorflow implementation Yandong Wen's paper with a novel contrastive loss function
[link](http://ydwen.github.io/papers/WenECCV16.pdf).

In this paper, Wen *et.al.* attempt to achieve stronger inter-class differentiation by using a 
novel loss function that penalizes intra-class spread. Here I attempt to replicate some of the 
results with a LeNet implemented in Tensorflow.

## Learnings

- **2D feature embedding** The paper uses a very nice 2-D embedding for the feature layer. This makes an interesting 
visualization, and can be seen in this figure bellow. Each color corresponds to a different 
class label. 

<img width="400" height="322" alt="2 D Deep Features" src="LeNet_plus_centerloss/figures/MNIST%20LeNet%2B%2B%20with%202%20Deep%20Features%20(PReLU).png"/>

- **Usage of PReLU** The paper uses PReLU activation function. At the time I used Xavier initialization,
but He initialization would be more appropriate.

- **Evolution of embedding** Curious about the evolution of the network, I made a few videos showing
how the embedding evolves during training. You can find a few here [here](LeNet_plus_centerloss/figures/animation/)

<p align="center">
    ![network learning](LeNet_plus_centerloss/figures/animation/MNIST_LeNet_centroid_loss_lambda_0.001.gif)
</p>

## Usage

To **train** run:
```shell
python3 ./LeNet_plus_centerloss/MNIST_train.py
```

To **evaluate** run:
```shell
python3 ./LeNet_plus_centerloss/MNIST_eval.py
```

### Prerequisite

Have Tensorflow installed, with GPU support. 

**NOTE** In order for the eval script to run at the same time as the 
training, you would need to have multiple devices (GPUs), or share the
same CPU (will be slow).
