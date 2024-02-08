> Disclaimer


# Introduction


# Methodology

## Metrics

* Proving time
* Memory usage
* Accuracy loss

## Frameworks

## Models

It is worthig noting that the exisiting ONNX framework used to convert tensorflow to pytorch or verse wise is not working most of time, therefore I mannually define two exactly same NN network and transfer the weights and biases based on each special way to compute, for example, weight in pytorch will be transposed first and then used to multiple the input matrix. Therefore, our benchmark tool not only tests the performance results regards to the proving time and memory usage, but also provide more reasonable results on the accuracy loss between two different packages that zkml frameworks based on.

# Results

## DNN


## CNN

# Analysis