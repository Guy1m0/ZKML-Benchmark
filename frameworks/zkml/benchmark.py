import numpy as np
import re, os, argparse, sys
import tensorflow as tf
import concurrent.futures, subprocess, threading, psutil, time
import pandas as pd

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

def get_predictions(interpreter, test_images):
    predictions = []

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i, img in enumerate(test_images):
        test_image = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Set the value for the input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)
        
        # Run the inference
        interpreter.invoke()

        # Retrieve the output and dequantize
        output = interpreter.get_tensor(output_details[0]['index'])
        output = np.argmax(output, axis=1)
        predicted_class = output[0]

        predictions.append(predicted_class)


    return predictions

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
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(monitor_memory, process.pid)
        stdout, stderr = process.communicate()
        max_memory = future.result()
    if show:
        print(f"Maximum memory used: {max_memory} MB")
        print("Total time:", time.time() - start_time)
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
    
def benchmark(test_images, predictions, model_name, model_in_path, circuit_folder, test = False, save = False, notes = ""):
    # Convert the model
    tmp_folder = "./tmp/"
    msgpack_folder = tmp_folder + "msgpack/"
    os.makedirs(msgpack_folder, exist_ok=True)

    model_convert_path = "./tools/converter.py"
    model_out_path = tmp_folder + "msgpack/converted_model.msgpack"
    config_path = tmp_folder + "msgpack/config.msgpack"

    scale_factor = 512
    k = 17
    num_cols = 10 
    num_randoms = 1024 

    command = ["python", model_convert_path, "--model", f"{model_in_path}",
            "--model_output", f"{model_out_path}", "--config_output",
            config_path, "--scale_factor", str(scale_factor),
            "--k", str(k), "--num_cols", str(num_cols), "--num_randoms",
            str(num_randoms)]
    execute_and_monitor(command)


    loss = 0
    img_inputs_path = tmp_folder + "inputs/"
    os.makedirs(img_inputs_path, exist_ok=True)
    
    input_convert_path = "./tools/input_converter.py"
    config_out_path = msgpack_folder+"config.msgpack"

    time_circuit = circuit_folder + "time_circuit"
    test_circuit = circuit_folder + "test_circuit"

    call_path = time_circuit
    if test:
        call_path = test_circuit

    model_out_path = tmp_folder + "msgpack/converted_model.msgpack"

    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i, img in enumerate(test_images):
        cost = 0
        print ("Process for image", i)
        start_time = time.time()

        np.save(f"{img_inputs_path}{str(i)}.npy", img)
        
        # Convert the input to the model
        img_in_path = img_inputs_path + str(i)+ ".npy"
        img_out_path = msgpack_folder + "img_" + str(i) + ".msgpack"


        command_1 = ["python", f"{input_convert_path}", "--model_config", f"{config_out_path}",
                "--inputs", img_in_path, "--output", img_out_path]
        # print (command_1)
        command_2 = [call_path, model_out_path, img_out_path, "kzg"]
        # print (command_2)
        _, _, usage = execute_and_monitor(command_1)
        cost += usage
        stdout, _, usage = execute_and_monitor(command_2)
        cost += usage

        # Extract x values using regex
        x_values = [int(x) for x in re.findall(r'final out\[\d+\] x: (-?\d+) \(', stdout)][-10:]
        #x_values = [int(x) for x in re.findall(r'final out\[\d+\] x: (\d+)', stdout)][-10:]
        #print (x_values)

        # Find max value and its index
        max_value = max(x_values)
        max_index = x_values.index(max_value)
        # print (max_index)
        
        if max_index != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", max_index)
        
        mem_usage.append(cost)
        time_cost.append(time.time() - start_time)

    print ("Total time:", time.time() - benchmark_start_time)
    layers = model_name.split("_")

    if int(layers[0]) < 30:
        arch = arch_folders[model_name][:-1]
        arch = '-'.join(word.capitalize() for word in arch.split('-')) + '_Kernal'
        layers[0] = str(int(layers[0])**2)

        new_row = {
            'Framework': ['zkml (tensorflow)'],
            'Architecture': [f'{arch} ({"x".join(layers[:-1])}_{layers[-1]}x{layers[-1]})'],
            '# Layers': [len(layers)-1],
            '# Parameters': [params[model_name]],
            'Testing Size': [len(mem_usage)],
            'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
            'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
            'Std Memory Usage': [pd.Series(mem_usage).std()],
            'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
            'Std Proving Time': [pd.Series(time_cost).std()],
            'Notes': notes
        }
        # arch = f'{arch_folder} ({"x".join(layers)})'
    else:
        layers = model_name.split("_")
        arch = "Input" + (len(layers)-1) * "-Dense"

        new_row = {
            'Framework': ['zkml (tensorflow)'],
            'Architecture': [f'{arch} ({"x".join(layers)})'],
            '# Layers': [len(layers)],
            '# Parameters': [params[model_name]],
            'Testing Size': [len(mem_usage)],
            'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
            'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
            'Std Memory Usage': [pd.Series(mem_usage).std()],
            'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
            'Std Proving Time': [pd.Series(time_cost).std()],
            'Notes': notes
        }

    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def load_csv():
    csv_path = '../../benchmarks/benchmark_results.csv'

    columns = ['Framework', 'Architecture', '# Layers', '# Parameters', 'Testing Size', 'Accuracy Loss (%)', 
            'Avg Memory Usage (MB)', 'Std Memory Usage', 'Avg Proving Time (s)', 'Std Proving Time']

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

def show_models():
    for key in params:
        layers = key.split("_")
        if int(layers[0]) < 30:
            arch = arch_folders[key]
        else:
            arch = "input" + (len(layers)-1) * "-dense" 

        print (f'model_name: {key} | arch: {arch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark result for a given model and testsize.",
        epilog="Example usage: python benchmark.py --size 100 --model model_name"
    )    
    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--agg', type=int, help='Set the start for aggregating benchmark results')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')
    parser.add_argument('--arm', action='store_true', help='Flag to indicate if use Arm64 Arch')


    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')

    if args.model not in params:
        print ("Please check the model name by using '--list'")
        sys.exit()

    layers = [int(x) for x in args.model.split("_")]
    model_path = "../../models/"

    if not args.save:
        args.save = False

    start = 0
    notes = ""
    if args.agg:
        start = args.agg
        notes = f'start from {start}'

    if layers[0] > 30:
        dnn = True
    else:
        dnn = False

    if args.arm:
        circuit_folder = "./arm_64/"
    else:
        circuit_folder = "./x86_64/"


    if dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"

        model_path = "../../models/"

        model_in_path = model_path+arch_folder+args.model + '.tflite'

        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()

        if layers[0] == 784:
            tests, _ = dnn_datasets()
        else:
            _, tests = dnn_datasets()
        predicted_labels = get_predictions(interpreter, tests)

    else:
        arch_folder = arch_folders[args.model]
        model_in_path = model_path + arch_folder + args.model +'.tflite'

        interpreter = tf.lite.Interpreter(model_path=model_in_path)
        interpreter.allocate_tensors()

        if layers[0] == 28:
            tests, _ = cnn_datasets()
        else:
            _, tests = cnn_datasets()
        predicted_labels = get_predictions(interpreter, tests)

    benchmark(tests[start:start+args.size], predicted_labels[start:start+args.size], args.model, model_in_path, circuit_folder, save=args.save, notes = notes)