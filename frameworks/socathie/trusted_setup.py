import os, subprocess, psutil, concurrent, time, re, argparse

def find_digit(output):
    match = re.search(r'non-linear constraints: (\d+)', output)
    if match:
        constraints = int(match.group(1))
        # Calculate k such that 2**k > 2 * constraints
        k = 1
        while 2**k <= 2 * constraints:
            k += 1
        print(f"Constraints: {constraints}, k: {k}")
        return k
    else:
        print("Constraints not found")

def setup(digit, model_name, output_folder, show = True):
    ceremony_folder = output_folder + 'ceremony/'
    os.makedirs(ceremony_folder, exist_ok=True)
    ptau_1 = ceremony_folder + 'pot12_0000.ptau'

    command = ['snarkjs', 'powersoftau', 'new', 'bn128', str(digit), ptau_1,'-v']
    print (command)
    subprocess.run(command)


    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    command = ["snarkjs", "powersoftau", "contribute", ptau_1, ptau_2, "--name=1st", "-v"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    process.communicate(input="abcd\n")

    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    ptau_3 = ceremony_folder + 'pot12_final.ptau'

    command = ['snarkjs', 'powersoftau', 'prepare', 'phase2', ptau_2,ptau_3, '-v']

    subprocess.run(command)


    r1cs_path = output_folder + model_name + ".r1cs"
    zkey_1 = ceremony_folder + 'test_0000.zkey'
    command = ['snarkjs', 'groth16', 'setup', r1cs_path, ptau_3, zkey_1]
    print (command)
    subprocess.run(command)


    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    zkey_2 = ceremony_folder + "test_0001.zkey"

    command = ["snarkjs", "zkey", "contribute", zkey_1, zkey_2, "--name=usr1", "-v"]
    print (command)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    process.communicate(input="1234\n")

    veri_key = ceremony_folder + 'vk.json'
    command = ['snarkjs', 'zkey', 'export','verificationkey', zkey_1, veri_key]
    print (command)
    subprocess.run(command)


    return zkey_1, veri_key



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--output', type=str, required=False, help='If save results')
    parser.add_argument('--show', type=bool, required=False, help='If show command detail')
    args = parser.parse_args()

    if not args.show:
        args.show = False

    circuit_folder = "./golden_circuits/"
    target_circom = args.model + '.circom'    
    output_folder = f'./{args.output}/'
    os.makedirs(output_folder, exist_ok=True)

    command = ['circom', circuit_folder + target_circom, "--r1cs", "--wasm", "--sym", "-o", output_folder]
    res = subprocess.run(command, capture_output=True, text = True)
    digit = find_digit(res.stdout)

    setup(digit, args.model, output_folder, args.show)

