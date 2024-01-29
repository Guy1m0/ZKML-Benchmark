# MNIST Example for GGML

This is a simple example of how to use GGML for inferencing.

## Training the Model

A notebook for training a simple two-layer network to recognize digits is located at `trainning/mnist.ipynb`. You can
use this to save a pytorch model to be converted to ggml format.



## GGML Format Conversion

GGML "format" is whatever you choose for efficient loading. In our case, we just save the hyperparameters used
plus the model weights and biases. Run convert-h5-to-ggml.py to convert your pytorch model. The output format is:

- magic constant (int32)
- repeated list of tensors
- number of dimensions of tensor (int32)
- tensor dimension (int32 repeated)
- values of tensor (int32)

Run ```convert-h5-to-ggml.py mnist_model.state_dict``` where `mnist_model.state_dict` is the saved pytorch model from the notebook. 

## MNIST Network

The MNIST recognizer network is extremely simple. A fully connected layer + relu, followed by a fully connected layer + softmax. This
version of the MNIST network doesn't use convolutions.

