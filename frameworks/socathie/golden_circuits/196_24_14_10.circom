pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/Dense.circom";
include "../node_modules/circomlib-ml/circuits/ReLU.circom";

template dnn() {
    signal input in[196];

    signal input Dense32weights[196][24];
    signal input Dense32bias[24];
    signal input Dense32out[24];
    signal input Dense32remainder[24];

    signal input ReLUout[24];

    signal input Dense21weights[24][14];
    signal input Dense21bias[14];
    signal input Dense21out[14];
    signal input Dense21remainder[14];

    signal input ReLUout2[14];

    signal input Dense10weights[14][10];
    signal input Dense10bias[10];
    signal input Dense10out[10];
    signal input Dense10remainder[10];

    signal output out[10];

    component Dense32 = Dense(196,24, 10**36);
    component relu[24];
    component Dense21 = Dense(24,14, 10**36);
    component relu2[14];
    component Dense10 = Dense(14,10, 10**36);


    for (var i=0; i<196; i++) {
        Dense32.in[i] <== in[i];
        for (var j=0; j<24; j++) {
            Dense32.weights[i][j] <== Dense32weights[i][j];
        }
    }

    for (var i=0; i<24; i++) {
        Dense32.bias[i] <== Dense32bias[i];
        Dense32.out[i] <== Dense32out[i];
        Dense32.remainder[i] <== Dense32remainder[i];
    }

    for (var i=0; i<24; i++) {
        relu[i] = ReLU();
        relu[i].in <== Dense32.out[i];
        relu[i].out <== ReLUout[i];
    }

    for (var i=0; i<24; i++) {
        Dense21.in[i] <== relu[i].out;
        for (var j=0; j<14; j++) {
            Dense21.weights[i][j] <== Dense21weights[i][j];
        }
    }

    for (var i=0; i<14; i++) {
        Dense21.bias[i] <== Dense21bias[i];
        Dense21.out[i] <== Dense21out[i];
        Dense21.remainder[i] <== Dense21remainder[i];        
    }

    for (var i=0; i<14; i++) {
        relu2[i] = ReLU();
        relu2[i].in <== Dense21.out[i];
        relu2[i].out <== ReLUout2[i];
    }

    for (var i=0; i<14; i++) {
        Dense10.in[i] <== relu2[i].out;
        for (var j=0; j<10; j++) {
            Dense10.weights[i][j] <== Dense10weights[i][j];
        }
    }

    for (var i=0; i<10; i++) {
        Dense10.bias[i] <== Dense10bias[i];
        Dense10.out[i] <== Dense10out[i];
        Dense10.remainder[i] <== Dense10remainder[i];
    }


    for (var i=0; i<10; i++) {
        out[i] <== Dense10.out[i];
    }
    
}

component main = dnn();