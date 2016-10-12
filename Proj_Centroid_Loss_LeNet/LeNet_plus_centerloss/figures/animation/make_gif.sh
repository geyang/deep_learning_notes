#!/usr/bin/env bash

convert -delay 5 -resize 20% -page lambda_3/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_centroid_loss_lambda_3.gif
convert -delay 5 -resize 20% -page lambda_1/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_centroid_loss_lambda_1.gif
convert -delay 5 -resize 20% -page lambda_0.1/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_centroid_loss_lambda_0.1.gif
convert -delay 5 -resize 20% -page lambda_0.01/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_centroid_loss_lambda_0.01.gif
convert -delay 5 -resize 20% -page lambda_0.001/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_centroid_loss_lambda_0.001.gif
convert -delay 5 -resize 20% -page pre_train_lambda_0.01/*png -loop 0 -coalesce -deconstruct -monitor MNIST_LeNet_pretrain_lambda_0.01.gif
