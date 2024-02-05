pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Conv2D.circom";
include "../node_modules/circomlib-ml/circuits/Flatten2D.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";
include "../node_modules/circomlib-ml/circuits/AveragePooling2D.circom";
include "../node_modules/circomlib-ml/circuits/Dense.circom";

template Model() {
signal input in[14][14][1];
signal input conv2d_157_weights[3][3][1][5];
signal input conv2d_157_bias[5];
signal input conv2d_157_out[12][12][5];
signal input conv2d_157_remainder[12][12][5];
signal input re_lu_136_out[12][12][5];
signal input average_pooling2d_156_out[6][6][5];
signal input average_pooling2d_156_remainder[6][6][5];
signal input conv2d_158_weights[3][3][5][11];
signal input conv2d_158_bias[11];
signal input conv2d_158_out[4][4][11];
signal input conv2d_158_remainder[4][4][11];
signal input re_lu_137_out[4][4][11];
signal input average_pooling2d_157_out[2][2][11];
signal input average_pooling2d_157_remainder[2][2][11];
signal input flatten_78_out[44];
signal input dense_176_weights[44][84];
signal input dense_176_bias[84];
signal input dense_176_out[84];
signal input dense_176_remainder[84];
signal input dense_177_weights[84][10];
signal input dense_177_bias[10];
signal input dense_177_out[10];
signal input dense_177_remainder[10];
signal input dense_178_weights[10][3];
signal input dense_178_bias[3];
signal input dense_178_out[3];
signal output out[3];
signal input dense_178_remainder[3];

component conv2d_157 = Conv2D(14, 14, 1, 5, 3, 1, 10**18);
component re_lu_136[12][12][5];
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            re_lu_136[i0][i1][i2] = ReLU();
}}}
component average_pooling2d_156 = AveragePooling2D(12, 12, 5, 2, 2);
component conv2d_158 = Conv2D(6, 6, 5, 11, 3, 1, 10**18);
component re_lu_137[4][4][11];
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            re_lu_137[i0][i1][i2] = ReLU();
}}}
component average_pooling2d_157 = AveragePooling2D(4, 4, 11, 2, 2);
component flatten_78 = Flatten2D(2, 2, 11);
component dense_176 = Dense(44, 84, 10**18);
component dense_177 = Dense(84, 10, 10**18);
component dense_178 = Dense(10, 3, 10**18);

for (var i0 = 0; i0 < 14; i0++) {
    for (var i1 = 0; i1 < 14; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            conv2d_157.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 1; i2++) {
            for (var i3 = 0; i3 < 5; i3++) {
                conv2d_157.weights[i0][i1][i2][i3] <== conv2d_157_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 5; i0++) {
    conv2d_157.bias[i0] <== conv2d_157_bias[i0];
}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_157.out[i0][i1][i2] <== conv2d_157_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_157.remainder[i0][i1][i2] <== conv2d_157_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            re_lu_136[i0][i1][i2].in <== conv2d_157.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            re_lu_136[i0][i1][i2].out <== re_lu_136_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d_156.in[i0][i1][i2] <== re_lu_136[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d_156.out[i0][i1][i2] <== average_pooling2d_156_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            average_pooling2d_156.remainder[i0][i1][i2] <== average_pooling2d_156_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 6; i0++) {
    for (var i1 = 0; i1 < 6; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            conv2d_158.in[i0][i1][i2] <== average_pooling2d_156.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 5; i2++) {
            for (var i3 = 0; i3 < 11; i3++) {
                conv2d_158.weights[i0][i1][i2][i3] <== conv2d_158_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 11; i0++) {
    conv2d_158.bias[i0] <== conv2d_158_bias[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_158.out[i0][i1][i2] <== conv2d_158_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            conv2d_158.remainder[i0][i1][i2] <== conv2d_158_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            re_lu_137[i0][i1][i2].in <== conv2d_158.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            re_lu_137[i0][i1][i2].out <== re_lu_137_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_157.in[i0][i1][i2] <== re_lu_137[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_157.out[i0][i1][i2] <== average_pooling2d_157_out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            average_pooling2d_157.remainder[i0][i1][i2] <== average_pooling2d_157_remainder[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 11; i2++) {
            flatten_78.in[i0][i1][i2] <== average_pooling2d_157.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 44; i0++) {
    flatten_78.out[i0] <== flatten_78_out[i0];
}
for (var i0 = 0; i0 < 44; i0++) {
    dense_176.in[i0] <== flatten_78.out[i0];
}
for (var i0 = 0; i0 < 44; i0++) {
    for (var i1 = 0; i1 < 84; i1++) {
        dense_176.weights[i0][i1] <== dense_176_weights[i0][i1];
}}
for (var i0 = 0; i0 < 84; i0++) {
    dense_176.bias[i0] <== dense_176_bias[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    dense_176.out[i0] <== dense_176_out[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    dense_176.remainder[i0] <== dense_176_remainder[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    dense_177.in[i0] <== dense_176.out[i0];
}
for (var i0 = 0; i0 < 84; i0++) {
    for (var i1 = 0; i1 < 10; i1++) {
        dense_177.weights[i0][i1] <== dense_177_weights[i0][i1];
}}
for (var i0 = 0; i0 < 10; i0++) {
    dense_177.bias[i0] <== dense_177_bias[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_177.out[i0] <== dense_177_out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_177.remainder[i0] <== dense_177_remainder[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    dense_178.in[i0] <== dense_177.out[i0];
}
for (var i0 = 0; i0 < 10; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        dense_178.weights[i0][i1] <== dense_178_weights[i0][i1];
}}
for (var i0 = 0; i0 < 3; i0++) {
    dense_178.bias[i0] <== dense_178_bias[i0];
}
for (var i0 = 0; i0 < 3; i0++) {
    dense_178.out[i0] <== dense_178_out[i0];
}
for (var i0 = 0; i0 < 3; i0++) {
    dense_178.remainder[i0] <== dense_178_remainder[i0];
}
for (var i0 = 0; i0 < 3; i0++) {
    out[i0] <== dense_178.out[i0];
}

}

component main = Model();
