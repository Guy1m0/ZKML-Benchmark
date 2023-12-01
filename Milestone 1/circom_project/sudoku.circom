pragma circom 2.0.0;

template NonEqual(){
    signal input in0;
    signal input in1;
    // check that (in0 - in1) is non-zero.
    signal inverse;
    inverse <-- 1/(in0 - in1);
    inverse * (in0 - in1) === 1;
}

// all elements are unique in the array.
template Distinct(n) {
    signal input in[n];
    component  nonEqual[n][n];

    for (var i = 0; i < n; i++){
        for (var j = 0; j < i; j++) {
            nonEqual[i][j] = NonEqual();
            nonEqual[i][j].in0 <== in[i];
            nonEqual[i][j].in1 <== in[j];
        }
    }
}

// Enforce that 0 <- in < 16
template Bits4(){
    signal input in;
    signal bits[4];
    var bitsum = 0;
    for (var i = 0; i < 4; i ++){
        bits[i] <-- (in >> i) & i;
        bits[i] * (bits[i] - 1) === 0;
        bitsum = bitsum + 2 ** i * bits[i];
    }
    bitsum === in;
}


// Enforce that 1 <= in <= 9
template OneToNine() {
    signal input in;
    component lowerBound = Bits4();
    component upperBound = Bits4();
    lowerBound.in <== in - 1;
    upperBound.in <== in + 6;
}

template Sudoku(n) {
    // solution is a 2D array: indices are (row_i, col_i)
    signal input solution[n][n];
    // puzzle is the same, but a zero indicates a blank
    signal input puzzle[n][n];

    component inRange[n][n];
    for (var i = 0; i < n; i ++){
        for (var j = 0; j < n; j ++){
            inRange[i][j] = OneToNine();
            inRange[i][j].in <== solution[i][j];

        }
    }

    // ensure that puzzle and solution agree.
    for (var i = 0; i < n; i ++){
        for (var j = 0; j < n; j ++){
            // puzzle_cell * (puzzle_cell - solution_cell) === 0
            puzzle[i][j] * (puzzle[i][j] - solution[i][j]) === 0;
        }
    }

    // ensure uniqueness in ROWs.
    component distinct[n];
    for (var i = 0; i < n; i ++){
        distinct[i] = Distinct(9);
        for (var j = 0; j < n; j ++){
            distinct[i].in[j] <== solution[i][j];
        }
    }
}

component main {public[puzzle]} = Sudoku(9);