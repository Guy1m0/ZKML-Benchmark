# MLGO

MLGO is tensor library for machine learning in pure Golang that can run on MIPS.

The machine learning part of this project refers to the legendary [ggml.cpp](https://github.com/ggerganov/ggml) framework.


## Build

`pip install -r requirements.txt`

## MNIST

1. Train the AI model. See `examples/mnist/trainning/mnist.ipynb`
2. Convert the AI model into GGML using `examples/mnist/convert-h5-to-ggml.py`
3. Build the AI inference engine for MIPS
`cd examples/mnist_mips && ./build`
``

## LLAMA

Support LLaMA 1 and LLaMA 2 (7B and 13B).
Please refer to `examples/llama/README.md` and download the model first.