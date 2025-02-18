pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Flatten2D.circom";
include "../node_modules/circomlib-ml/circuits/AveragePooling2D.circom";
include "../node_modules/circomlib-ml/circuits/Conv2D.circom";
include "../node_modules/circomlib-ml/circuits/Dense.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";

template Model() {
signal input in[14][14][1];
signal input conv2d_weights[3][3][1][5];
signal input conv2d_bias[5];
signal input conv2d_out[12][12][5];
signal input conv2d_remainder[12][12][5];

signal input conv2d_re_lu_out[12][12][5];

signal input average_pooling2d_out[6][6][5];
signal input average_pooling2d_remainder[6][6][5];

signal input conv2d_1_weights[3][3][5][11];
signal input conv2d_1_bias[11];
signal input conv2d_1_out[4][4][11];
signal input conv2d_1_remainder[4][4][11];

signal input conv2d_1_re_lu_out[4][4][11];

signal input average_pooling2d_1_out[2][2][11];
signal input average_pooling2d_1_remainder[2][2][11];

signal input flatten_out[44];

signal input dense_weights[44][80];
signal input dense_bias[80];
signal input dense_out[80];
signal input dense_remainder[80];

signal input dense_re_lu_out[80];

signal input dense_1_weights[80][10];
signal input dense_1_bias[10];
signal input dense_1_out[10];

signal output out[10];
signal input dense_1_remainder[10];

component conv2d = Conv2D(14, 14, 1, 5, 3, 1, 10**18);
component conv2d_re_lu[12][12][5];
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_re_lu[i0][i1][i2] = ReLU();
}}}
component average_pooling2d = AveragePooling2D(12, 12, 5, 2, 2);
component conv2d_1 = Conv2D(6, 6, 5, 11, 3, 1, 10**18);
component conv2d_1_re_lu[4][4][11];
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_1_re_lu[i0][i1][i2] = ReLU();
}}}
component average_pooling2d_1 = AveragePooling2D(4, 4, 11, 2, 2);
component flatten = Flatten2D(2, 2, 11);
component dense = Dense(44, 80, 10**18);
component dense_1 = Dense(80, 10, 10**18);

component dense_relu[80];

for (var i0 = 0; i0 < 14; i0++) {
    for (var i1 = 0; i1 < 14; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            conv2d.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            for (var i3 = 0; i3 < 5; i3++) {
                conv2d.weights[i0][i1][i2][i3] <== conv2d_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 5; i0++) {
    conv2d.bias[i0] <== conv2d_bias[i0];
}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d.out[i0][i1][i2] <== conv2d_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d.remainder[i0][i1][i2] <== conv2d_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_re_lu[i0][i1][i2].in <== conv2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_re_lu[i0][i1][i2].out <== conv2d_re_lu_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d.in[i0][i1][i2] <== conv2d_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d.out[i0][i1][i2] <== average_pooling2d_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d.remainder[i0][i1][i2] <== average_pooling2d_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_1.in[i0][i1][i2] <== average_pooling2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            for (var i3 = 0; i3 < 11; i3++) {
                conv2d_1.weights[i0][i1][i2][i3] <== conv2d_1_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 11; i0++) {
    conv2d_1.bias[i0] <== conv2d_1_bias[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_1.out[i0][i1][i2] <== conv2d_1_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_1.remainder[i0][i1][i2] <== conv2d_1_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_1_re_lu[i0][i1][i2].in <== conv2d_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_1_re_lu[i0][i1][i2].out <== conv2d_1_re_lu_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_1.in[i0][i1][i2] <== conv2d_1_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_1.out[i0][i1][i2] <== average_pooling2d_1_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_1.remainder[i0][i1][i2] <== average_pooling2d_1_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            flatten.in[i0][i1][i2] <== average_pooling2d_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 44; i0++) {
    flatten.out[i0] <== flatten_out[i0];
}
for (var i0 = 0; i0 < 44; i0++) {
    dense.in[i0] <== flatten.out[i0];
}
for (var i0 = 0; i0 < 44; i0++) {
    for (var i1 = 0; i1 < 80; i1++) {
        dense.weights[i0][i1] <== dense_weights[i0][i1];
}}
for (var i0 = 0; i0 < 80; i0++) {
    dense.bias[i0] <== dense_bias[i0];
}
for (var i0 = 0; i0 < 80; i0++) {
    dense.out[i0] <== dense_out[i0];
}
for (var i0 = 0; i0 < 80; i0++) {
    dense.remainder[i0] <== dense_remainder[i0];
}

for (var i0 = 0; i0 < 80; i0++) {
    dense_relu[i0] = ReLU();
    dense_relu[i0].in <== dense.out[i0];
    dense_relu[i0].out <== dense_re_lu_out[i0];
}


for (var i0 = 0; i0 < 80; i0++) {
    dense_1.in[i0] <== dense_relu[i0].out;
    for (var i1 = 0; i1 < 10; i1++) {
        dense_1.weights[i0][i1] <== dense_1_weights[i0][i1];
}}

for (var i0 = 0; i0 < 10; i0++) {
    dense_1.bias[i0] <== dense_1_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_1.out[i0] <== dense_1_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_1.remainder[i0] <== dense_1_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    out[i0] <== dense_1.out[i0];
}

}

component main = Model();
