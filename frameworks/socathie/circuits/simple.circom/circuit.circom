pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/ArgMax.circom";
include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[25];
signal input dense_34_weights[25][10];
signal input dense_34_bias[10];
signal input dense_34_out[10];
signal input dense_34_remainder[10];
signal input dense_35_weights[10][2];
signal input dense_35_bias[2];
signal input dense_35_out[2];
signal input dense_35_remainder[2];
signal input softmax_13_out[1];
signal output out[1];

component dense_34 = Dense(25, 10, 10**18);
component dense_35 = Dense(10, 2, 10**18);
component softmax_13 = ArgMax(2);

for (var i0 = 0; i0 < 25; i0++) {
    dense_34.in[i0] <== in[i0];
}
for (var i0 = 0; i0 < 25; i0++) {
    for (var i1 = 0; i1 < 10; i1++) {
        dense_34.weights[i0][i1] <== dense_34_weights[i0][i1];
}}
for (var i0 = 0; i0 < 10; i0++) {
    dense_34.bias[i0] <== dense_34_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_34.out[i0] <== dense_34_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_34.remainder[i0] <== dense_34_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_35.in[i0] <== dense_34.out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        dense_35.weights[i0][i1] <== dense_35_weights[i0][i1];
}}
for (var i0 = 0; i0 < 2; i0++) {
    dense_35.bias[i0] <== dense_35_bias[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    dense_35.out[i0] <== dense_35_out[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    dense_35.remainder[i0] <== dense_35_remainder[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    softmax_13.in[i0] <== dense_35.out[i0];
}
for (var i0 = 0; i0 < 1; i0++) {
    softmax_13.out[i0] <== softmax_13_out[i0];
}
out[0] <== softmax_13.out;

}

component main = Model();
