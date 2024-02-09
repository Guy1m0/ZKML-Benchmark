import os, subprocess, psutil, concurrent.futures, time, re, argparse, sys

import pandas as pd

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

import subprocess
import concurrent.futures
import psutil
import time

def monitor_memory(pid, freq=0.01):
    p = psutil.Process(pid)
    max_memory = 0
    while True:
        try:
            mem = p.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
            max_memory = max(max_memory, mem)
        except psutil.NoSuchProcess:
            break  # Process has finished
        time.sleep(freq)  # Poll at the specified frequency
    return max_memory

def execute_and_monitor(command, show=False):
    # Start the subprocess without PIPE to allow stdout and stderr to print directly to console
    process = subprocess.Popen(command, stdout=None, stderr=None, text=True)

    # Start monitoring memory in a separate thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        
        # Wait for the process to complete
        while process.poll() is None:
            time.sleep(0.1)  # Sleep to avoid busy waiting
        
        max_memory = future.result()
    
    if show:
        print(f"Maximum memory used: {max_memory} MB")
    return max_memory

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')

def extract_info(output):
    match = re.search(r'non-linear constraints: (\d+)', output)
    if not match:
        print("Constraints not found")
        return
    
    constraints = int(match.group(1))
    # Calculate k such that 2**k > 2 * constraints
    k = 1
    while 2**k <= 2 * constraints:
        k += 1
    print(f"Constraints: {constraints}, k: {k}")

    return constraints, k
    
        

def setup(digit, model_name, output_folder):
    mem_usage = 0
    wrapper_path = './snarkjs_wrapper.js'
    start_time = time.time()
    ceremony_folder = output_folder + f'{str(digit)}/'
    os.makedirs(ceremony_folder, exist_ok=True)
    ptau_1 = ceremony_folder + 'pot12_0000.ptau'

    command = ['snarkjs', 'powersoftau', 'new', 'bn128', str(digit), ptau_1,'-v']
    print (command)
    mem_usage = max(mem_usage, execute_and_monitor(command))



    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    command = ["snarkjs", "powersoftau", "contribute", ptau_1, ptau_2, "--name=1st", "-v"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    process.communicate(input="abcd\n")

    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    ptau_3 = ceremony_folder + 'pot12_final.ptau'

    command = ['snarkjs', 'powersoftau', 'prepare', 'phase2', ptau_2,ptau_3, '-v']

    mem_usage = max(mem_usage, execute_and_monitor(command))


    r1cs_path = output_folder + model_name + ".r1cs"
    zkey_1 = ceremony_folder + 'test_0000.zkey'
    command = ['node', wrapper_path, 'groth16', 'setup', r1cs_path, ptau_3, zkey_1]
    print (command)
    mem_usage = max(mem_usage, execute_and_monitor(command))


    ptau_2 = ceremony_folder + 'pot12_0001.ptau'
    zkey_2 = ceremony_folder + "test_0001.zkey"

    command = ['snarkjs', "zkey", "contribute", zkey_1, zkey_2, "--name=usr1", "-v"]
    print (command)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    process.communicate(input="1234\n")

    veri_key = ceremony_folder + 'vk.json'
    command = ['snarkjs', 'zkey', 'export','verificationkey', zkey_1, veri_key]
    print (command)
    mem_usage = max(mem_usage, execute_and_monitor(command))

    return time.time() - start_time, mem_usage



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the provided model, generate trusted setup for later benchmarking")
        # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    # parser.add_argument('--digit', type =int, help='Specify the max support circuit size 2**digit')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--output', type=str, default="tmp",help='Specify the output folder')
    
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
    print (res.stdout)
    
    constraints, digit = extract_info(res.stdout)
    time_cost, mem_cost = setup(digit, args.model, output_folder)

    # Specify the CSV file name
    csv_path = "model_info.csv"

    # Define the CSV header (column names)
    csv_columns = ["Model Name", "Non-Linear Constraints", "Digits", "Time Cost", "Memory Required"]

    # Check if the CSV file exists
    if not os.path.isfile(csv_path):
        # Create a DataFrame with the specified columns
        df = pd.DataFrame(columns=csv_columns)
        # Save the DataFrame as a CSV file
        df.to_csv(csv_path, index=False)
    else:
        print(f"File '{csv_path}' already exists.")

    df = pd.read_csv(csv_path)

    new_row = {
        "Model Name": [args.model], 
        "Non-Linear Constraints": [constraints], 
        "Digits": [digit], 
        "Time Cost": [time_cost], 
        "Memory Required": [mem_cost]
    }

    new_row_df = pd.DataFrame(new_row)
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_path, index=False)


