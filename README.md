# mnist-nn-matlab
Implemented a three-layer NN with MATLAB, trained on MNIST dataset.

## Train

First, download dataset from http://yann.lecun.com/exdb/mnist/ and decompress it into ./dataset

Then, you can run ```train_model.m``` to train the network, it will automatically train, test and save the best model into model.mat

## Inference

First of all, load the best model from ```model.mat```,

```octave
load model
```

You may use ```predict_from_file(dist,W1, b1, W2, b2)``` function to get results.

## Note

I have used ```dlarray``` from MATLAB 2020a, so it may not works so well with lower version.
However, if you want to run it on lower version, 
you can delete the transformation procedures between original array and dlarray, like ```X = dlarray(X);``` and ```extractdata()```.

This is just an experiment I wrote before, as an assignment from class. I won't update it anymore.
