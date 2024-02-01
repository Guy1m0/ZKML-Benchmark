""" Make an interger-only circuit of the corresponding CIRCOM circuit.

Usage:
    circuit.py <circuit.json> <input.json> [-o <output>]
    circuit.py (-h | --help)

Options:
    -h --help                               Show this screen.
    -o <output> --output=<output>           Output directory [default: output].

"""

from docopt import docopt
import json

try:
    from keras2circom.util import *
except:
    import sys
    import os
    # add parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from keras2circom.util import *

def inference(input, circuit):
    out = input['in']
    output = {}
    
    out, remainder = DenseInt(25, 10, 10**18, out, circuit['dense_34_weights'], circuit['dense_34_bias'])
    output['dense_34_out'] = out
    output['dense_34_remainder'] = remainder

    out, remainder = DenseInt(10, 2, 10**18, out, circuit['dense_35_weights'], circuit['dense_35_bias'])
    output['dense_35_out'] = out
    output['dense_35_remainder'] = remainder

    out = ArgMaxInt(out)
    output['softmax_13_out'] = out


    return out, output


def main():
    """ Main entry point of the app """
    args = docopt(__doc__)
    
    # parse input.json
    with open(args['<input.json>']) as f:
        input = json.load(f)
    
    # parse circuit.json
    with open(args['<circuit.json>']) as f:
        circuit = json.load(f)

    out, output = inference(input, circuit)

    # write output.json
    with open(args['--output'] + '/output.json', 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
