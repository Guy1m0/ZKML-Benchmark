pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/ArgMax.circom";
include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[784];
signal input dense_42_weights[784][56];
signal input dense_42_bias[56];
signal input dense_42_out[56];
signal input dense_42_remainder[56];
signal input dense_43_weights[56][10];
signal input dense_43_bias[10];
signal input dense_43_out[10];
signal input dense_43_remainder[10];
signal input dense_43_softmax_out[1];
signal output out[1];

component dense_42 = Dense(784, 56, 10**18);
component dense_43 = Dense(56, 10, 10**18);
component dense_43_softmax = ArgMax(10);

for (var i0 = 0; i0 < 784; i0++) {
    dense_42.in[i0] <== in[i0];
}
for (var i0 = 0; i0 < 784; i0++) {
    for (var i1 = 0; i1 < 56; i1++) {
        dense_42.weights[i0][i1] <== dense_42_weights[i0][i1];
}}
for (var i0 = 0; i0 < 56; i0++) {
    dense_42.bias[i0] <== dense_42_bias[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_42.out[i0] <== dense_42_out[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_42.remainder[i0] <== dense_42_remainder[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_43.in[i0] <== dense_42.out[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    for (var i1 = 0; i1 < 10; i1++) {
        dense_43.weights[i0][i1] <== dense_43_weights[i0][i1];
}}
for (var i0 = 0; i0 < 10; i0++) {
    dense_43.bias[i0] <== dense_43_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_43.out[i0] <== dense_43_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_43.remainder[i0] <== dense_43_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_43_softmax.in[i0] <== dense_43.out[i0];
}
dense_43_softmax.out <== dense_43_softmax_out[0];
out[0] <== dense_43_softmax.out;

}

component main = Model();
