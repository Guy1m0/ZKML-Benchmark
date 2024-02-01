pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[25];
signal input dense_64_weights[25][10];
signal input dense_64_bias[10];
signal input dense_64_out[10];
signal input dense_64_remainder[10];
signal input dense_65_weights[10][4];
signal input dense_65_bias[4];
signal input dense_65_out[4];
signal output out[4];
signal input dense_65_remainder[4];

component dense_64 = Dense(25, 10, 10**18);
component dense_65 = Dense(10, 4, 10**18);

for (var i0 = 0; i0 < 25; i0++) {
    dense_64.in[i0] <== in[i0];
}
for (var i0 = 0; i0 < 25; i0++) {
    for (var i1 = 0; i1 < 10; i1++) {
        dense_64.weights[i0][i1] <== dense_64_weights[i0][i1];
}}
for (var i0 = 0; i0 < 10; i0++) {
    dense_64.bias[i0] <== dense_64_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_64.out[i0] <== dense_64_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_64.remainder[i0] <== dense_64_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_65.in[i0] <== dense_64.out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        dense_65.weights[i0][i1] <== dense_65_weights[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    dense_65.bias[i0] <== dense_65_bias[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    dense_65.out[i0] <== dense_65_out[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    dense_65.remainder[i0] <== dense_65_remainder[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    out[i0] <== dense_65.out[i0];
}

}

component main = Model();
