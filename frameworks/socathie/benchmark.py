import numpy as np
import tensorflow as tf

import subprocess, os, argparse, sys, re
import concurrent.futures, json, threading, psutil, time

import pandas as pd


p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

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

def get_predictions_tf(model, test_images, batch_size=256):
    predictions = []
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
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

def Conv2DInt(nRows, nCols, nChannels, nFilters, kernelSize, strides, n, input, weights, bias):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    Weights = [[[[str(weights[i][j][k][l] % p) for l in range(nFilters)] for k in range(nChannels)] for j in range(kernelSize)] for i in range(kernelSize)]
    Bias = [str(bias[i] % p) for i in range(nFilters)]
    out = [[[0 for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    remainder = [[[None for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    for i in range((nRows - kernelSize)//strides + 1):
        for j in range((nCols - kernelSize)//strides + 1):
            for m in range(nFilters):
                for k in range(nChannels):
                    for x in range(kernelSize):
                        for y in range(kernelSize):
                            out[i][j][m] += input[i*strides+x][j*strides+y][k] * weights[x][y][k][m]
                out[i][j][m] += bias[m]
                remainder[i][j][m] = str(out[i][j][m] % n)
                out[i][j][m] = str(out[i][j][m] // n % p)
    return Input, Weights, Bias, out, remainder

def AveragePooling2DInt (nRows, nCols, nChannels, poolSize, strides, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [[[0 for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    remainder = [[[None for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    for i in range((nRows-poolSize)//strides + 1):
        for j in range((nCols-poolSize)//strides + 1):
            for k in range(nChannels):
                for x in range(poolSize):
                    for y in range(poolSize):
                        out[i][j][k] += input[i*strides+x][j*strides+y][k]
                remainder[i][j][k] = str(out[i][j][k] % poolSize**2 % p)
                out[i][j][k] = str(out[i][j][k] // poolSize**2 % p)
    return Input, out, remainder

def prepare_input_json_cnn(X, layers, input_path):
    kernal_size = layers[-1]
    X_in = [[[int(X[i][j][0]*1e18)] for j in range(layers[0])] for i in range(layers[0])]
    conv2d_1_weights = [[[[int(model.layers[1].weights[0][i][j][k][l]*1e18) for l in range(layers[1])] for k in range(1)] for j in range(kernal_size)] for i in range(kernal_size)]
    conv2d_1_bias = [int(model.layers[1].weights[1][i]*1e36) for i in range(layers[1])]

    X_in, conv2d_1_weights, conv2d_1_bias, conv2d_1_out, conv2d_1_remainder = Conv2DInt(layers[0], layers[0], 1, layers[1], kernal_size, 1, 10**18, X_in, conv2d_1_weights, conv2d_1_bias)

    relu_1_in = [[[int(conv2d_1_out[i][j][k]) for k in range(5)] for j in range(12)] for i in range(12)]
    relu_1_out = [[[str(relu_1_in[i][j][k]) if relu_1_in[i][j][k] < p//2 else 0 for k in range(5)] for j in range(12)] for i in range(12)]

    avg2d_1_in = [[[int(relu_1_out[i][j][k]) for k in range(5)] for j in range(12)] for i in range(12)]

    _, avg2d_1_out, avg2d_1_remainder = AveragePooling2DInt(12, 12, 5, 2, 2, avg2d_1_in)


    conv2d_2_in = [[[int(avg2d_1_out[i][j][k]) for k in range(5)] for j in range(6)] for i in range(6)]
    conv2d_2_weights = [[[[int(model.layers[4].weights[0][i][j][k][l]*1e18) for l in range(11)] for k in range(5)] for j in range(kernal_size)] for i in range(kernal_size)]
    conv2d_2_bias = [int(model.layers[4].weights[1][i]*1e36) for i in range(11)]

    _, conv2d_2_weights, conv2d_2_bias, conv2d_2_out, conv2d_2_remainder = Conv2DInt(6, 6, 5, 11, 3, 1, 10**18, conv2d_2_in, conv2d_2_weights, conv2d_2_bias)


    relu_2_in = [[[int(conv2d_2_out[i][j][k]) for k in range(11)] for j in range(4)] for i in range(4)]
    relu_2_out = [[[str(relu_2_in[i][j][k]) if relu_2_in[i][j][k] < p//2 else 0 for k in range(11)] for j in range(4)] for i in range(4)]


    avg2d_2_in = [[[int(relu_2_out[i][j][k]) if int(relu_2_out[i][j][k]) < p//2 else int(relu_2_out[i][j][k]) - p for k in range(11)] for j in range(4)] for i in range(4)]

    _, avg2d_2_out, avg2d_2_remainder = AveragePooling2DInt(4, 4, 11, 2, 2, avg2d_2_in)

    flatten_out = [avg2d_2_out[i][j][k] for i in range(2) for j in range(2) for k in range(11)]

    dense_in = [int(flatten_out[i]) if int(flatten_out[i]) < p//2 else int(flatten_out[i]) - p for i in range(44)]
    dense_weights = [[int(model.layers[8].weights[0][i][j]*1e18) for j in range(80)] for i in range(44)]
    dense_bias = [int(model.layers[8].weights[1][i]*1e36) for i in range(80)]

    _, dense_weights, dense_bias, dense_out, dense_remainder = DenseInt(44, 80, 10**18, dense_in, dense_weights, dense_bias)

    dense_1_weights = [[int(model.layers[10].weights[0][i][j]*1e18) for j in range(10)] for i in range(80)]
    dense_1_bias = [int(model.layers[10].weights[1][i]*1e36) for i in range(10)]

    re_lu_3_out = [dense_out[i] if int(dense_out[i]) < p//2 else 0 for i in range(80)]
    dense_1_in = [int(re_lu_3_out[i]) for i in range(80)]

    _, dense_1_weights, dense_1_bias, dense_1_out, dense_1_remainder = DenseInt(80, 10, 10**18, dense_1_in, dense_1_weights, dense_1_bias)

    in_json = {
        "in": X_in,
        "conv2d_2_weights": conv2d_1_weights,
        "conv2d_2_bias": conv2d_1_bias,
        "conv2d_2_out": conv2d_1_out,
        "conv2d_2_remainder": conv2d_1_remainder,
        "re_lu_1_out": relu_1_out,
        "average_pooling2d_1_out": avg2d_1_out,
        "average_pooling2d_1_remainder": avg2d_1_remainder,
        "conv2d_3_weights": conv2d_2_weights,
        "conv2d_3_bias": conv2d_2_bias,
        "conv2d_3_out": conv2d_2_out,
        "conv2d_3_remainder": conv2d_2_remainder,
        "re_lu_2_out": relu_2_out,
        "average_pooling2d_2_out": avg2d_2_out,
        "average_pooling2d_2_remainder": avg2d_2_remainder,
        "flatten_out": flatten_out,
        "dense_weights": dense_weights,
        "dense_bias": dense_bias,
        "dense_out": dense_out,
        "dense_remainder": dense_remainder,
        "re_lu_3_out":re_lu_3_out,
        "dense_1_weights":dense_1_weights,
        "dense_1_bias":dense_1_bias,
        "dense_1_out":dense_1_out,
        "dense_1_remainder":dense_1_remainder
    }

    out = [int(x) for x in dense_1_out]
    out = [x if x < p//2 else 0 for x in out]
    pred = np.argmax(out)

    with open(input_path, "w") as f:
        json.dump(in_json, f)

    return pred

def prepare(model, layers):
    if layers[0] == 196:
        _, test_images = dnn_datasets()
    elif layers[0] == 784:
        test_images, _ = dnn_datasets()

    predictions_tf = get_predictions_tf(model, test_images)

    return predictions_tf, test_images

def cnn_datasets():
    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images_tf = train_images / 255.0
    test_images_tf = test_images / 255.0
    train_images_tf = train_images_tf.reshape(train_images.shape[0], 28, 28, 1)
    test_images_tf = test_images_tf.reshape(test_images.shape[0], 28, 28, 1)

    train_images_tf_14 = tf.image.resize(train_images_tf, [14, 14]).numpy()
    test_images_tf_14 = tf.image.resize(test_images_tf, [14, 14]).numpy()

    return test_images_tf, test_images_tf_14

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
def gen_model_cnn(layers, model_in_path):
    kernal_size = layers[-1]

    # Define the LeNet model in TensorFlow
    inputs = tf.keras.layers.Input(shape=(layers[0],layers[0],1))
    out = tf.keras.layers.Conv2D(layers[1],kernal_size, use_bias = True)(inputs)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.AveragePooling2D()(out)
    out = tf.keras.layers.Conv2D(layers[2],kernal_size, use_bias = True)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.AveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(layers[3])(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dense(layers[4])(out)
    model = tf.keras.Model(inputs, out)

    model.load_weights(model_in_path)

    # Compile the model
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

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
        print ("process for image ",i)
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
        # print ('wit:', wit_json_file)
        # print ('input:', input_path)
        # print ('target:', target_circom)
        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove', zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]
                    #['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']]

        for command in commands:
            stdout, _, usage = execute_and_monitor(command)
            if "ERROR" in stdout:
                #print ('command:', command)

                print (stdout)
                return
            cost += usage
            # subprocess.run(command)
        #print ("stdout:", stdout)
            
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)
    
    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'{arch_folder} ({"x".join(layers)})'],
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
    wit_file = tmp_folder + target_circom[:-7] + "_witness.wtns"

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        cost = 0
        X = test_images[i:i+1]
        print ("process for image ",i)
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
            stdout, _, usage = execute_and_monitor(command)
            if "ERROR" in stdout:
                #print ('command:', command)
                print (stdout)
                return
            cost += usage
        #print ("stdout:", stdout)
            
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)
    
    print ("Total time:", time.time() - benchmark_start_time)
    layers = model_name.split("_")
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'{arch_folder} ({"x".join(layers)})'],
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

def benchmark_cnn(test_images, predictions, layers, model_name, tmp_folder, input_path, zkey, veri_key, save=False):
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
        print ("process for image ",i)
        cost = 0
        X = test_images[i]
        start_time = time.time()
        pred = prepare_input_json_cnn(X, layers, input_path)

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i)

        commands = [['node', wit_json_file, wasm_file, input_path, wit_file],
                    ['snarkjs', 'groth16', 'prove',zkey, wit_file, tmp_folder+'proof.json', tmp_folder+'public.json']]
                    #['snarkjs', 'groth16', 'verify',veri_key, tmp_folder+'public.json', tmp_folder+'proof.json']]

        for command in commands:
            stdout, _, usage = execute_and_monitor(command)
            if "ERROR" in stdout:
                print (stdout)
                return
            cost += usage
        #print ("stdout:", stdout)
            
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)

    print ("Total time:", time.time() - benchmark_start_time)
    layers = model_name.split("_")
    arch_folder = arch_folders[model_name]

    layers = layers[:-1]
    layers[0] = str(int(layers[0])**2)
    new_row = {
        'Framework': ['circomlib-ml (tensorflow)'],
        'Architecture': [f'{arch_folder} ({"x".join(layers)})'],
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

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')

def update_zkey(ceremony_folder, model_name):
    print ('Update zkey to avoid mismatch in witness and circuit')
    r1cs_path = output_folder + model_name + ".r1cs"
    zkey_1 = ceremony_folder + 'test_0000.zkey'
    ptau_3 = ceremony_folder + 'pot12_final.ptau'
    command = ['snarkjs', 'groth16', 'setup', r1cs_path, ptau_3, zkey_1]
    #print (command)
    res = subprocess.run(command, capture_output=True, text = True)
    if "ERROR" in res.stdout:
        print (res.stdout)



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")

    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')

    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')

    parser.add_argument('--output', type=str, default="tmp",help='Specify the output folder')

    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if args.model not in params:
        print ("Please check the model name by using '--list'")
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')

    layers = [int(x) for x in args.model.split("_")]
    model_path = "../../models/"

    output_folder = f'./{args.output}/'
    os.makedirs(output_folder, exist_ok=True)
    target_circom = "_".join(str(x) for x in layers) + '.circom'

    command = ['circom', "./golden_circuits/" + target_circom, "--r1cs", "--wasm", "--sym", "-o", output_folder]
    #command = ['circom', circuit_folder + target_circom, "--r1cs", "--wasm", "--sym", "-o", output_folder]
    res = subprocess.run(command, capture_output=True, text = True)
    print (res.stdout)
    digit = find_digit(res.stdout)

    zkey_1 = output_folder + f"{str(digit)}/test_0000.zkey"
    veri_key = output_folder + f"{str(digit)}/vk.json"

    # Check if zkey_1 and veri_key exist
    if not os.path.exists(zkey_1) or not os.path.exists(veri_key):
        print ('Start Trusted-setup before performing benchmark')
        # Call trusted_setup.py if either file does not exist
        trusted_setup_command = [
            'python', 'trusted_setup.py',
            '--model', args.model,
            '--output', args.output
        ]
        subprocess.run(trusted_setup_command)

    # update_zkey(output_folder + str(digit)+"/", args.model)
    # subprocess.run(command)
    # sys.exit()
    # _, _, _ = execute_and_monitor(command)

    if layers[0] > 30:
        dnn = True
    else:
        dnn = False

    if dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.h5'

        model = gen_model_dnn(layers, model_in_path)

        predicted_labels, tests = prepare(model, layers)
        weights, biases = transfer_weights(layers, model, 36)

        if len(layers)==3:
            benchmark(tests[:args.size], predicted_labels[:args.size], weights, biases,
                  layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, save=args.save)
        if len(layers)==4:
            #print ("other")
            benchmark_(tests[:args.size], predicted_labels[:args.size], weights, biases,
                  layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, save=args.save)
    else:
        arch_folder = arch_folders[args.model]
        model_path = "../../models/"
        model_in_path = model_path+arch_folder+args.model + '.h5'

        model = gen_model_cnn(layers, model_in_path)

        _, tests = cnn_datasets()

        predicted_labels = get_predictions_tf(model, tests)

        benchmark_cnn(tests[:args.size], predicted_labels[:args.size], 
                layers, args.model, output_folder, output_folder+"input.json", zkey_1, veri_key, save=args.save)
