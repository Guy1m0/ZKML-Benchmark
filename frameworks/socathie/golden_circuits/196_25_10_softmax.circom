pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";
include "../node_modules/circomlib-ml/circuits/ArgMax.circom";

template dnn() {
    signal input in[196];

    signal input Dense32weights[196][25];
    signal input Dense32bias[25];
    signal input Dense32out[25];
    signal input Dense32remainder[25];

    signal input ReLUout[25];

    signal input Dense21weights[25][10];
    signal input Dense21bias[10];
    signal input Dense21out[10];
    signal input Dense21remainder[10];

    signal input argmax_out;

    signal output out;

    component Dense32 = Dense(196,25, 10**36);
    component relu[25];
    component Dense21 = Dense(25,10, 10**36);
    component argmax = ArgMax(10);

    for (var i=0; i<196; i++) {
        Dense32.in[i] <== in[i];
        for (var j=0; j<25; j++) {
            Dense32.weights[i][j] <== Dense32weights[i][j];
        }
    }

    for (var i=0; i<25; i++) {
        Dense32.bias[i] <== Dense32bias[i];
        Dense32.out[i] <== Dense32out[i];
        Dense32.remainder[i] <== Dense32remainder[i];
    }

    for (var i=0; i<25; i++) {
        relu[i] = ReLU();
        relu[i].in <== Dense32.out[i];
        relu[i].out <== ReLUout[i];
    }

    for (var i=0; i<25; i++) {
        Dense21.in[i] <== relu[i].out;
        for (var j=0; j<10; j++) {
            Dense21.weights[i][j] <== Dense21weights[i][j];
        }
    }

    for (var i=0; i<10; i++) {
        Dense21.bias[i] <== Dense21bias[i];
        Dense21.out[i] <== Dense21out[i];
        Dense21.remainder[i] <== Dense21remainder[i];        
    }
    for (var i=0; i<10; i++) {
        argmax.in[i] <== Dense21.out[i];
    }

    argmax.out <== argmax_out;
    out <== argmax.out
    
}

component main = dnn();