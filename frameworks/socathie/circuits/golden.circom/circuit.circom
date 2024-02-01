pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[784];
signal input dense_54_weights[784][56];
signal input dense_54_bias[56];
signal input dense_54_out[56];
signal input dense_54_remainder[56];
signal input dense_55_weights[56][10];
signal input dense_55_bias[10];
signal input dense_55_out[10];
signal output out[10];
signal input dense_55_remainder[10];

component dense_54 = Dense(784, 56, 10**18);
component dense_55 = Dense(56, 10, 10**18);

for (var i0 = 0; i0 < 784; i0++) {
    dense_54.in[i0] <== in[i0];
}
for (var i0 = 0; i0 < 784; i0++) {
    for (var i1 = 0; i1 < 56; i1++) {
        dense_54.weights[i0][i1] <== dense_54_weights[i0][i1];
}}
for (var i0 = 0; i0 < 56; i0++) {
    dense_54.bias[i0] <== dense_54_bias[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_54.out[i0] <== dense_54_out[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_54.remainder[i0] <== dense_54_remainder[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    dense_55.in[i0] <== dense_54.out[i0];
}
for (var i0 = 0; i0 < 56; i0++) {
    for (var i1 = 0; i1 < 10; i1++) {
        dense_55.weights[i0][i1] <== dense_55_weights[i0][i1];
}}
for (var i0 = 0; i0 < 10; i0++) {
    dense_55.bias[i0] <== dense_55_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_55.out[i0] <== dense_55_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_55.remainder[i0] <== dense_55_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    out[i0] <== dense_55.out[i0];
}

}

component main = Model();
