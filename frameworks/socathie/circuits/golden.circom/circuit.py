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
    
    out, remainder = DenseInt(784, 56, 10**18, out, circuit['dense_54_weights'], circuit['dense_54_bias'])
    output['dense_54_out'] = out
    output['dense_54_remainder'] = remainder

    out, remainder = DenseInt(56, 10, 10**18, out, circuit['dense_55_weights'], circuit['dense_55_bias'])
    output['dense_55_out'] = out
    output['dense_55_remainder'] = remainder


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
