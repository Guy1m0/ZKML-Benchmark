import os, subprocess, psutil, concurrent, time, re, argparse, sys

params = {"784_56_10": 44543,
            "196_25_10": 5185,
            "196_24_14_10": 5228,
            "28_6_16_10_5":5142,
            "14_5_11_80_10_3": 4966}

accuracys = {"784_56_10": 0.9740,
            "196_25_10": 0.9541,
            "196_24_14_10": 0.9556,
            "14_5_11_80_10_3": 0.9556}

arch_folders = {"28_6_16_10_5": "input-conv2d-conv2d-dense/",
                "14_5_11_80_10_3": "input-conv2d-conv2d-dense-dense/",
                "28_6_16_120_84_10_5": "input-conv2d-conv2d-dense-dense-dense/"}

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')

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

def setup(digit, model_name, output_folder):
    start_time = time.time()
    ceremony_folder = output_folder + f'{str(digit)}/'
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
    
    print ('Total time cost:', time.time() - start_time)

    return zkey_1, veri_key



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the provided model, generate trusted setup for later benchmarking")
        # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    # parser.add_argument('--digit', type =int, help='Specify the max support circuit size 2**digit')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--output', type=str, default="./tmp/",help='Specify the output folder')
    
    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if args.model is None:
        parser.error('--model is required for trusted setup.')

    circuit_folder = "./golden_circuits/"
    target_circom = args.model + '.circom'    
    output_folder = f'./{args.output}/'
    os.makedirs(output_folder, exist_ok=True)

    command = ['circom', circuit_folder + target_circom, "--r1cs", "--wasm", "--sym", "-o", output_folder]
    res = subprocess.run(command, capture_output=True, text = True)
    digit = find_digit(res.stdout)

    setup(digit, args.model, output_folder)

