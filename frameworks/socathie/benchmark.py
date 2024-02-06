import numpy as np
import tensorflow as tf
import subprocess

import subprocess, os, argparse
import concurrent.futures, json, threading, psutil, time

import pandas as pd


p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

params = {"784_56_10": 44543,
          "196_25_10": 5185,
          "196_24_14_10": 5228,
            "28_6_16_10_5": 5142,
            "14_5_11_80_10_3": 4966, # @TODO: May doublecheck
            "28_6_16_120_84_10_5": 44530}

accuracys = {"784_56_10": 0.9740,
            "196_25_10": 0.9541,
            "196_24_14_10": 0.9556,
            "28_6_16_10_5": 0.9877,
            "14_5_11_80_10_3": 0.9556, # @TODO: May doublecheck
            "28_6_16_120_84_10_5": 0.9877}

arch_folders = {"28_6_16_10_5": "input-conv2d-conv2d-dense/",
                "14_5_11_80_10_3": "input-conv2d-conv2d-dense-dense/",
                "28_6_16_120_84_10_5": "input-conv2d-conv2d-dense-dense-dense/"}

def get_predictions_tf(model, test_images, batch_size=256):
    predictions = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        pred = model.predict(batch)
        predictions.extend(np.argmax(pred, axis=1))
    return predictions

def transfer_weights(layers, model, scalar = 36):
    weights = []
    biases = []
    for ind in range(len(layers)-1):
        w = [[int(model.weights[ind * 2].numpy()[i][j]*10**scalar) for j in range(layers[ind+1])] for i in range(layers[ind])]
        b = [int(model.weights[ind * 2 + 1].numpy()[i]*10**(scalar * 2)) for i in range(layers[ind+1])]
        #b = [0 for i in range(layers[ind+1])]
        weights.append(w)
        biases.append(b)

    return weights, biases

def relu_mod(x):
    return x if x < p // 2 else 0

def DenseInt(nInputs, nOutputs, n, input, weights, bias):
    #print (len(input), nInputs)
    
    Input = [str(input[i] % p) for i in range(nInputs)]
    Weights = [[str(weights[i][j] % p) for j in range(nOutputs)] for i in range(nInputs)]
    Bias = [str(bias[i] % p) for i in range(nOutputs)]
    
    out = [0 for _ in range(nOutputs)]
    remainder = [None for _ in range(nOutputs)]
    
    for j in range(nOutputs):
        for i in range(nInputs):
            out[j] += input[i] * weights[i][j]
        out[j] += bias[j]

        remainder[j] = str(out[j] % n)
        out[j] = out[j] // n % p
        
    return Input, Weights, Bias, out, remainder

def prepare_input_json(layers, weights, biases, x_in, scalar=36, relu = False):
    relu_outs = []
    dense_weights = []
    dense_biases = []
    dense_outs = []
    dense_remainders = []
    x_ins = []

    out = x_in
    for ind in range(len(weights)):
        nInputs = layers[ind]
        nOutputs = layers[ind + 1]
        #print (nInputs, nOutputs)
        x_in, w, b, out, rem = DenseInt(nInputs, nOutputs, 10 ** scalar, 
                                     out, weights[ind], biases[ind])
        
        dense_outs.append(out)
        if relu:
            out = [x if x < p//2 else 0 for x in out]
            relu_outs.append([str(x) if x !=0 else 0 for x in out ])

        #print (out)
        dense_weights.append(w)
        dense_biases.append(b)
        
        dense_remainders.append(rem)
        x_ins.append(x_in)

    
    dense_outs = [[str(x) for x in sub] for sub in dense_outs]
        
    return x_ins[0], dense_weights, dense_biases, dense_outs, dense_remainders, relu_outs, np.argmax(out)

def monitor_memory(pid, freq = 0.01):
    p = psutil.Process(pid)
    max_memory = 0
    while True:
        try:
            mem = p.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, mem)
        except psutil.NoSuchProcess:
            break  # Process has finished
        time.sleep(freq)  # Poll every second
        
    #print(f"Maximum memory used: {max_memory} MB")
    return max_memory

def execute_and_monitor(command, show = False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        stdout, stderr = process.communicate()
        max_memory = future.result()
    if show:
        print(f"Maximum memory used: {max_memory} MB")
    return stdout, stderr, max_memory

def dnn_datasets():
    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and flatten the images
    train_images_tf = train_images.reshape((-1, 28*28)) / 255.0
    test_images_tf = test_images.reshape((-1, 28*28)) / 255.0

    # Resize for 14 * 14 images
    train_images_tf_reshaped = tf.reshape(train_images_tf, [-1, 28, 28, 1])  # Reshape to [num_samples, height, width, channels]
    test_images_tf_reshaped = tf.reshape(test_images_tf, [-1, 28, 28, 1])

    # Downsample images
    train_images_tf_downsampled = tf.image.resize(train_images_tf_reshaped, [14, 14], method='bilinear')
    test_images_tf_downsampled = tf.image.resize(test_images_tf_reshaped, [14, 14], method='bilinear')

    # Flatten the images back to [num_samples, 14*14]
    train_images_tf_downsampled = tf.reshape(train_images_tf_downsampled, [-1, 14*14])
    test_images_tf_downsampled = tf.reshape(test_images_tf_downsampled, [-1, 14*14])

    return test_images_tf, test_images_tf_downsampled

def prepare(layers):
    if layers[0] == 196:
        _, test_images = dnn_datasets()
    elif layers[0] == 784:
        test_images, _ = dnn_datasets()

    predictions_tf = get_predictions_tf(model, test_images)

    return predictions_tf, test_images

def cnn_datasets():
    print ()

def gen_model_dnn(layers, model_in_path):
    if len(layers) == 3:
        inputs = tf.keras.layers.Input(shape=(layers[0],))
        out = tf.keras.layers.Dense(layers[1], activation = 'relu')(inputs)
        out = tf.keras.layers.Dense(layers[2])(out)

        model = tf.keras.Model(inputs, out)

    elif len(layers) == 4:
        inputs = tf.keras.layers.Input(shape=(layers[0],))
        out = tf.keras.layers.Dense(layers[1], activation = 'relu')(inputs)
        out = tf.keras.layers.Dense(layers[2], activation = 'relu')(out)
        out = tf.keras.layers.Dense(layers[3])(out)

        model = tf.keras.Model(inputs, out)
    else:
        print ("Layers not Support")
        return None
    
    model.load_weights(model_in_path)
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    return model

# @ TODO: Hardcoded
# def gen_model_cnn(layers, state_dict):
#     if len(layers) == 6:
        
            
#     elif len(layers) == 7:
        
#     elif len(layers) == 5:
        
#     else:
#         print ("Layers not Support")
#         return None
    

#     return model

def load_csv():
    csv_path = '../../benchmarks/benchmark_results.csv'

    columns = ['Framework', 'Architecture', '# Layers', '# Parameters', 'Testing Size', 'Accuracy Loss (%)', 
            'Avg Memory Usage (MB)', 'Std Memory Usage', 'Avg Proving Time (s)', 'Std Proving Time', 'Notes']

    # Check if the CSV file exists
    if not os.path.isfile(csv_path):
        # Create a DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        # Save the DataFrame as a CSV file
        df.to_csv(csv_path, index=False)
    else:
        print(f"File '{csv_path}' already exists.")

    df = pd.read_csv(csv_path)
    return df

def benchmark(test_images, predictions, weights, biases, layers, model_name, tmp_folder, input_path, zkey, veri_key, save=False):
    loss = 0

    target_circom = "_".join(str(x) for x in layers) + '.circom'

    json_folder = tmp_folder + target_circom[:-7] + "_js/"
    wit_json_file = json_folder + "generate_witness.js"
    wasm_file = json_folder + target_circom[:-7] + ".wasm"
    input_path = tmp_folder + "input.json"
    wit_file = tmp_folder + "witness.wtns"

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        cost = 0
        X = test_images[i:i+1]
        start_time = time.time()
        X_in = [int(x*1e36) for x in X[0]]
        x_in, dense_weights, dense_biases, dense_outs, dense_remainders, relu_outs, pred = prepare_input_json(layers, weights, biases, X_in, scalar=36, relu=True)

        in_json = {
            "in": x_in,
            "Dense32weights": dense_weights[0],
            "Dense32bias": dense_biases[0],
            "Dense32out": dense_outs[0],
            "Dense32remainder": dense_remainders[0],
            "ReLUout": relu_outs[0], 
            "Dense21weights": dense_weights[1],
            "Dense21bias": dense_biases[1],
            "Dense21out": dense_outs[1],
            "Dense21remainder": dense_remainders[1]
        }

        with open(input_path, "w") as f:
            json.dump(in_json, f)

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i)

        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove',zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]
                    #['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']]

        for command in commands:
            _, _, usage = execute_and_monitor(command)
            cost += usage
        #print ("stdout:", stdout)
            
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)
    
    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'Input-Dense-Dense ({"x".join(layers)})'],
        '# Layers': [len(layers)],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()]
    }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def benchmark_(test_images, predictions, weights, biases, layers, model_name, tmp_folder, input_path, zkey, veri_key, save=False):
    loss = 0

    target_circom = "_".join(str(x) for x in layers) + '.circom'

    json_folder = tmp_folder + target_circom[:-7] + "_js/"
    wit_json_file = json_folder + "generate_witness.js"
    wasm_file = json_folder + target_circom[:-7] + ".wasm"
    input_path = tmp_folder + "input.json"
    wit_file = tmp_folder + "witness.wtns"

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        cost = 0
        X = test_images[i:i+1]
        start_time = time.time()
        X_in = [int(x*1e36) for x in X[0]]
        x_in, dense_weights, dense_biases, dense_outs, dense_remainders, relu_outs, pred = prepare_input_json(layers, weights, biases, X_in, scalar=36, relu=True)


        in_json = {
            "in": x_in,
            "Dense32weights": dense_weights[0],
            "Dense32bias": dense_biases[0],
            "Dense32out": dense_outs[0],
            "Dense32remainder": dense_remainders[0],
            "ReLUout": relu_outs[0], 
            "Dense21weights": dense_weights[1],
            "Dense21bias": dense_biases[1],
            "Dense21out": dense_outs[1],
            "Dense21remainder": dense_remainders[1],
            "ReLUout2": relu_outs[1],
            "Dense10weights": dense_weights[2],
            "Dense10bias": dense_biases[2],
            "Dense10out": dense_outs[2],
            "Dense10remainder": dense_remainders[2]
        }

        with open(input_path, "w") as f:
            json.dump(in_json, f)

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i)

        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove',zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]
                    #['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']]

        for command in commands:
            _, _, usage = execute_and_monitor(command)
            cost += usage
        #print ("stdout:", stdout)
            
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)
    
    print ("Total time:", time.time() - benchmark_start_time)
    layers = model_name.split("_")
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'Input-Dense-Dense-Dense ({"x".join(layers)})'],
        '# Layers': [len(layers)],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()]
    }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")
    parser.add_argument('--size', type=int, required=True, help='Test Size')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--save', type=bool, required=False, help='If save results')
    parser.add_argument('--dnn', action='store_true', help='Flag to indicate if this is a DNN model')
    parser.add_argument('--cnn', action='store_false', dest='dnn', help='Flag to indicate if this is not a DNN model')
    parser.add_argument('--output', type=str, required=False, help='If save results')
    
    parser.add_argument('--accuracy', type=bool, required=False, help='Chose accuracy mode (CNN model not support)')

    args = parser.parse_args()

    args = parser.parse_args()
    layers = [int(x) for x in args.model.split("_")]
    model_path = "../../models/"

    output_folder = f'./{args.output}/'
    os.makedirs(output_folder, exist_ok=True)
    zkey_1 = output_folder + "ceremony/test_0000.zkey"
    veri_key = output_folder + "ceremony/vk.json"

    if not args.save:
        args.save = False

    if args.dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.h5'

        model = gen_model_dnn(layers, model_in_path)

        predicted_labels, tests = prepare(layers)
        weights, biases = transfer_weights(layers, model, 36)

        if len(layers)==3:
            benchmark(tests[:args.size], predicted_labels[:args.size], weights, biases,
                  layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, save=args.save)
        if len(layers)==4:
            print ("other")
            benchmark_(tests[:args.size], predicted_labels[:args.size], weights, biases,
                  layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, save=args.save)
    # else:
    #     arch_folder = arch_folders[args.model]
        
    #     state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

    #     model = gen_model_cnn(layers, state_dict)
        
    #     predicted_labels, tests = prepare_cnn(model, layers)

    #     benchmark_cnn(tests[:args.size], predicted_labels[:args.size], model, args.model, 
    #             save=args.save)