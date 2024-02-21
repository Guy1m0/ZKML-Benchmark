pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Conv2D.circom";
include "../node_modules/circomlib-ml/circuits/Flatten2D.circom";
include "../node_modules/circomlib-ml/circuits/AveragePooling2D.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";
include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[28][28][1];
signal input conv2d_weights[5][5][1][6];
signal input conv2d_bias[6];
signal input conv2d_out[24][24][6];
signal input conv2d_remainder[24][24][6];

signal input conv2d_re_lu_out[24][24][6];

signal input average_pooling2d_out[12][12][6];
signal input average_pooling2d_remainder[12][12][6];

signal input conv2d_1_weights[5][5][6][16];
signal input conv2d_1_bias[16];
signal input conv2d_1_out[8][8][16];
signal input conv2d_1_remainder[8][8][16];

signal input conv2d_1_re_lu_out[8][8][16];

signal input average_pooling2d_1_out[4][4][16];
signal input average_pooling2d_1_remainder[4][4][16];

signal input flatten_out[256];

signal input dense_weights[256][120];
signal input dense_bias[120];
signal input dense_out[120];
signal input dense_remainder[120];

signal input dense_re_lu_out[120];

signal input dense_1_weights[120][84];
signal input dense_1_bias[84];
signal input dense_1_out[84];
signal input dense_1_remainder[84];

signal input dense_1_re_lu_out[84];

signal input dense_2_weights[84][10];
signal input dense_2_bias[10];
signal input dense_2_out[10];
signal input dense_2_remainder[10];

signal output out[10];

component conv2d = Conv2D(28, 28, 1, 6, 5, 1, 10**18);
component conv2d_re_lu[24][24][6];
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d_re_lu[i0][i1][i2] = ReLU();
}}}
component average_pooling2d = AveragePooling2D(24, 24, 6, 2, 2);
component conv2d_1 = Conv2D(12, 12, 6, 16, 5, 1, 10**18);
component conv2d_1_re_lu[8][8][16];
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            conv2d_1_re_lu[i0][i1][i2] = ReLU();
}}}
component average_pooling2d_1 = AveragePooling2D(8, 8, 16, 2, 2);
component flatten = Flatten2D(4, 4, 16);
component dense = Dense(256, 120, 10**18);
component dense_relu[120];

component dense_1 = Dense(120, 84, 10**18);
component dense_1_relu[84];

component dense_2 = Dense(84, 10, 10**18);

for (var i0 = 0; i0 < 28; i0++) {
    for (var i1 = 0; i1 < 28; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            conv2d.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 5; i0++) {
    for (var i1 = 0; i1 < 5; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            for (var i3 = 0; i3 < 6; i3++) {
                conv2d.weights[i0][i1][i2][i3] <== conv2d_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 6; i0++) {
    conv2d.bias[i0] <== conv2d_bias[i0];
}
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d.out[i0][i1][i2] <== conv2d_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d.remainder[i0][i1][i2] <== conv2d_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d_re_lu[i0][i1][i2].in <== conv2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d_re_lu[i0][i1][i2].out <== conv2d_re_lu_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 24; i0++) {
    for (var i1 = 0; i1 < 24; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            average_pooling2d.in[i0][i1][i2] <== conv2d_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            average_pooling2d.out[i0][i1][i2] <== average_pooling2d_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            average_pooling2d.remainder[i0][i1][i2] <== average_pooling2d_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            conv2d_1.in[i0][i1][i2] <== average_pooling2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 5; i0++) {
    for (var i1 = 0; i1 < 5; i1++) {
        for (var i2 = 0; i2 < 6; i2++) {
            for (var i3 = 0; i3 < 16; i3++) {
                conv2d_1.weights[i0][i1][i2][i3] <== conv2d_1_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 16; i0++) {
    conv2d_1.bias[i0] <== conv2d_1_bias[i0];
}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            conv2d_1.out[i0][i1][i2] <== conv2d_1_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            conv2d_1.remainder[i0][i1][i2] <== conv2d_1_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            conv2d_1_re_lu[i0][i1][i2].in <== conv2d_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            conv2d_1_re_lu[i0][i1][i2].out <== conv2d_1_re_lu_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            average_pooling2d_1.in[i0][i1][i2] <== conv2d_1_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            average_pooling2d_1.out[i0][i1][i2] <== average_pooling2d_1_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            average_pooling2d_1.remainder[i0][i1][i2] <== average_pooling2d_1_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 16; i2++) {
            flatten.in[i0][i1][i2] <== average_pooling2d_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 256; i0++) {
    flatten.out[i0] <== flatten_out[i0];
}
for (var i0 = 0; i0 < 256; i0++) {
    dense.in[i0] <== flatten.out[i0];
}
for (var i0 = 0; i0 < 256; i0++) {
    for (var i1 = 0; i1 < 120; i1++) {
        dense.weights[i0][i1] <== dense_weights[i0][i1];
}}
for (var i0 = 0; i0 < 120; i0++) {
    dense.bias[i0] <== dense_bias[i0];
}
for (var i0 = 0; i0 < 120; i0++) {
    dense.out[i0] <== dense_out[i0];
}
for (var i0 = 0; i0 < 120; i0++) {
    dense.remainder[i0] <== dense_remainder[i0];
}

for (var i0 = 0; i0 < 120; i0++) {
    dense_relu[i0] = ReLU();
    dense_relu[i0].in <== dense.out[i0];
    dense_relu[i0].out <== dense_re_lu_out[i0];
}

for (var i0 = 0; i0 < 120; i0++) {
    dense_1.in[i0] <== dense_relu[i0].out;
    for (var i1 = 0; i1 < 84; i1++) {
        dense_1.weights[i0][i1] <== dense_1_weights[i0][i1];
}}

for (var i0 = 0; i0 < 84; i0++) {
    dense_1.bias[i0] <== dense_1_bias[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    dense_1.out[i0] <== dense_1_out[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    dense_1.remainder[i0] <== dense_1_remainder[i0];
}

for (var i0 = 0; i0 < 84; i0++) {
    dense_1_relu[i0] = ReLU();
    dense_1_relu[i0].in <== dense_1.out[i0];
    dense_1_relu[i0].out <== dense_1_re_lu_out[i0];
}


for (var i0 = 0; i0 < 84; i0++) {
    dense_2.in[i0] <== dense_1_relu[i0].out;
    for (var i1 = 0; i1 < 10; i1++) {
        dense_2.weights[i0][i1] <== dense_2_weights[i0][i1];
}}

for (var i0 = 0; i0 < 10; i0++) {
    dense_2.bias[i0] <== dense_2_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_2.out[i0] <== dense_2_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_2.remainder[i0] <== dense_2_remainder[i0];
}

for (var i0 = 0; i0 < 10; i0++) {
    out[i0] <== dense_2.out[i0];
}

}

component main = Model();
