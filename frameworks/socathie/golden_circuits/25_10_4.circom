pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";

template dnn() {
    signal input in[25];

    signal input Dense32weights[25][10];
    signal input Dense32bias[10];
    signal input Dense32out[10];
    signal input Dense32remainder[10];

    signal input ReLUout[10];

    signal input Dense21weights[10][4];
    signal input Dense21bias[4];
    signal input Dense21out[4];
    signal input Dense21remainder[4];

    signal output out[4];

    component Dense32 = Dense(25,10, 10**36);
    component relu[10];
    component Dense21 = Dense(10,4, 10**36);

    for (var i=0; i<25; i++) {
        Dense32.in[i] <== in[i];
        for (var j=0; j<10; j++) {
            Dense32.weights[i][j] <== Dense32weights[i][j];
        }
    }

    for (var i=0; i<10; i++) {
        Dense32.bias[i] <== 0;
        Dense32.out[i] <== Dense32out[i];
        Dense32.remainder[i] <== Dense32remainder[i];
    }

    for (var i=0; i<10; i++) {
        relu[i] = ReLU();
        relu[i].in <== Dense32.out[i];
        relu[i].out <== ReLUout[i];
    }

    for (var i=0; i<10; i++) {
        Dense21.in[i] <== relu[i].out;
        for (var j=0; j<4; j++) {
            Dense21.weights[i][j] <== Dense21weights[i][j];
        }
    }

    for (var i=0; i<4; i++) {
        Dense21.bias[i] <== Dense21bias[i];
        Dense21.out[i] <== Dense21out[i];
        Dense21.remainder[i] <== Dense21remainder[i];        
    }
    for (var i=0; i<4; i++) {
        out[i] <== Dense21.out[i];
    }
    
}

component main = dnn();