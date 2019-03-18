# C-implementation-of-RNN

## Introduction
This is the first step to implement RNN on FPGAs. All modules are heavily commented. We will use High-Level Synthesis to turn these code into Hardware Description Languages (HDL).

## Module Description
rnn.c: RNN Layers

fc.c: Fully-connected Layers 

softmax.c: Softmax Layers 


## H5dump

list all contents:
h5dump -n pre-trained-rnn.h5

open an attribute
h5dump -a "/model_weights/dense_1/weight_names" pre-trained-rnn.h5

view a dataset (show weights)
h5dump -d "/model_weights/dense_1/dense_1/bias:0" pre-trained-rnn.h5

h5 to txt:
h5dump -o dense_1_bias.txt -y -w 1000000000 dense_1_bias.h5
