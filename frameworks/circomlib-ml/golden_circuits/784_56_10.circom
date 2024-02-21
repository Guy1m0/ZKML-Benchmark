pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";

template dnn() {
    signal input in[784];

    signal input Dense32weights[784][56];
    signal input Dense32bias[56];
    signal input Dense32out[56];
    signal input Dense32remainder[56];

    signal input ReLUout[56];

    signal input Dense21weights[56][10];
    signal input Dense21bias[10];
    signal input Dense21out[10];
    signal input Dense21remainder[10];

    signal output out[10];

    component Dense32 = Dense(784,56, 10**36);
    component relu[56];
    component Dense21 = Dense(56,10, 10**36);

    for (var i=0; i<784; i++) {
        Dense32.in[i] <== in[i];
        for (var j=0; j<56; j++) {
            Dense32.weights[i][j] <== Dense32weights[i][j];
        }
    }

    for (var i=0; i<56; i++) {
        Dense32.bias[i] <== Dense32bias[i];
        Dense32.out[i] <== Dense32out[i];
        Dense32.remainder[i] <== Dense32remainder[i];
    }

    for (var i=0; i<56; i++) {
        relu[i] = ReLU();
        relu[i].in <== Dense32.out[i];
        relu[i].out <== ReLUout[i];
    }

    for (var i=0; i<56; i++) {
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
        out[i] <== Dense21.out[i];
    }
    
}

component main = dnn();