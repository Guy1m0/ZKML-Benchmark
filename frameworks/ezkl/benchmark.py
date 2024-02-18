import os, subprocess, sys
import torch, struct, os, psutil, subprocess, time, threading
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import json, ezkl
import pandas as pd

import subprocess, concurrent
import psutil
import time
import argparse

from torch.utils.data import DataLoader, TensorDataset

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


def dnn_datasets():
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Convert to PyTorch tensors
    test_images_pt = torch.tensor(test_images).float()
    test_labels_pt = torch.tensor(test_labels)
    # Flatten and normalize the images
    test_images_pt = test_images_pt.view(-1, 28*28) / 255.0  # Flatten and normalize

    # Assuming test_images_pt is your PyTorch tensor with shape [num_samples, 784]
    test_images_pt_reshaped = test_images_pt.view(-1, 1, 28, 28)  # Reshape to [num_samples, channels, height, width]

    # Downsample images
    test_images_pt_downsampled = F.interpolate(test_images_pt_reshaped, size=(14, 14), mode='bilinear', align_corners=False)

    # Flatten the images back to [num_samples, 14*14]
    test_images_pt_downsampled = test_images_pt_downsampled.view(-1, 14*14)

    return test_images_pt, test_images_pt_downsampled, test_labels_pt

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

    # Convert to PyTorch format [batch_size, channels, height, width]
    train_images_pt = torch.tensor(train_images_tf).permute(0, 3, 1, 2).float()
    test_images_pt = torch.tensor(test_images_tf).permute(0, 3, 1, 2).float()


    train_images_pt_14 =  torch.tensor(test_images_tf_14).permute(0, 3, 1, 2).float()
    test_images_pt_14 =  torch.tensor(test_images_tf_14).permute(0, 3, 1, 2).float()

    return test_images_pt, test_images_pt_14

def evaluate_pytorch_model(model, datasets, labels):
    # Create TensorDataset for test data
    test_dataset = TensorDataset(datasets, labels)
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


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

def benchmark_dnn(test_images, predictions, model, model_name, mode = "resources", output_folder='./tmp/', save = False, notes = ""):
    data_path = os.path.join(output_folder, 'input.json')
    model_path = os.path.join(output_folder, 'network.onnx')

    sampled_data = test_images[0]
    
    with torch.no_grad():
        torch.onnx.export(model, 
                    sampled_data, 
                    model_path, 
                    export_params=True, 
                    opset_version=10, 
                    do_constant_folding=True, 
                    input_names=['input_0'], 
                    output_names=['output'])
    loss = 0
    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i, img in enumerate(test_images):
        print ("Process for image", i)
        start_time = time.time()
        # Convert the tensor to numpy array and reshape it for JSON serialization
        x = (img.cpu().detach().numpy().reshape([-1])).tolist()
        data = dict(input_data = [x])

        # Serialize data into file:
        json.dump(data, open(data_path, 'w'))

        command = ["python", "gen_proof.py", "--model", model_path, "--data", data_path, "--output", output_folder, "--mode", mode]
        # subprocess.run(command)
        # stdout = "1234"
        # usage = 1
        stdout, _, usage = execute_and_monitor(command)

        try:
            pred = int(stdout[-2])
        except ValueError:
            print(f"Failed to convert {stdout[-2]} to int. Full output: {stdout}")
            pred  = -1
        #print ('pred:',pred)
        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", pred)
        mem_usage.append(usage)
        time_cost.append(time.time() - start_time)

    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    arch = "Input" + (len(layers)-1) * "-Dense"
    new_row = {
        'Framework': ['ezkl (pytorch)'],
        'Architecture': [f'{arch} ({"x".join(layers)})'],
        '# Layers': [len(layers)],
        '# Parameters': [params[model_name]],
        'Testing Size': [len(mem_usage)],
        'Accuracy Loss (%)': [loss/len(mem_usage) * 100],
        'Avg Memory Usage (MB)': [sum(mem_usage) / len(mem_usage)],
        'Std Memory Usage': [pd.Series(mem_usage).std()],
        'Avg Proving Time (s)': [sum(time_cost) / len(time_cost)],
        'Std Proving Time': [pd.Series(time_cost).std()],
        'Notes': [f'mode={mode}']
    }

    if notes:
        new_row['Notes'] = [new_row['Notes'][0] + " | " + notes]
    new_row_df = pd.DataFrame(new_row)
    print (new_row_df)

    if save:
        df = load_csv()
        df = pd.concat([df, new_row_df], ignore_index=True)
        csv_path = '../../benchmarks/benchmark_results.csv'
        df.to_csv(csv_path, index=False)

    return

def benchmark_cnn(test_images, predictions, model, model_name, mode = "resources", output_folder='./tmp/', save=False):
    data_path = os.path.join(output_folder, 'input.json')
    model_path = os.path.join(output_folder, 'network.onnx')

    loss = 0
    mem_usage = []
    time_cost = []
    benchmark_start_time = time.time()

    for i in range(len(test_images)):
        print ("Process for image", i)
        start_time = time.time()
        img = test_images[i:i+1]
        with torch.no_grad():
            torch.onnx.export(model, 
                        img, 
                        model_path, 
                        export_params=True, 
                        opset_version=10, 
                        do_constant_folding=True, 
                        input_names=['input_0'], 
                        output_names=['output'])
    
        command = ["python", "gen_proof.py", "--model", model_path, "--data", data_path, "--output", output_folder, "--mode", mode]
        # subprocess.run(command)
        # stdout = "1234"
        # usage = 1
        stdout, _, usage = execute_and_monitor(command)
        
        try:
            pred = int(stdout[-2])
        except ValueError:
            print(f"Failed to convert {stdout[-2]} to int. Full output: {stdout}")
            pred  = -1

        if pred != predictions[i]:
            loss += 1
            print ("Loss happens on index", i, "predicted_class", pred)
        mem_usage.append(usage)
        time_cost.append(time.time() - start_time)

    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    arch = arch_folders[model_name][:-1]
    arch = '-'.join(word.capitalize() for word in arch.split('-')) + '_Kernal'

    layers[0] = str(int(layers[0])**2)

    new_row = {
        'Framework': ['ezkl (pytorch)'],
        'Architecture': [f'{arch} ({"x".join(layers[:-1])}_{layers[-1]}x{layers[-1]})'],
        '# Layers': [len(layers)-1],
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

def gen_model_dnn(layers, state_dict):
    if len(layers) == 3:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

    elif len(layers) == 4:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])
                self.fc3 = nn.Linear(layers[2], num_classes)  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

# @ TODO: Hardcoded
def gen_model_cnn(layers, state_dict):
    if len(layers) == 6:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, layers[1], layers[-1]) 
                self.conv2 = nn.Conv2d(layers[1], layers[2], layers[-1]) 

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(11 * 2 * 2, layers[3]) # 256 * 120
                self.fc2 = nn.Linear(layers[3], layers[4])

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),layers[2],-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = self.fc2(x)  # No activation function here, will use CrossEntropyLoss later
                return x
            
    elif len(layers) == 7:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
                self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(16 *4 * 4,120) # 256 * 120
                self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
                self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),16,-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
                return x
    elif len(layers) == 5:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Convolutional encoder
                self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
                self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

                # Fully connected layers / Dense block
                self.fc1 = nn.Linear(16 *4 * 4,10) # 256 * 120

            def forward(self, x):
                # Convolutional block
                x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
                x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

                # TODO: figure out the resize, currently work on batch_size = 1
                batch_size = x.size(0)
                x = x.reshape(x.size(0),16,-1)  # 16 output channels
                x = np.transpose(x, (0,2,1)).reshape(batch_size,-1)
                #x = x.reshape(batch_size,-1)

                # Fully connected layers
                x = self.fc1(x)  # No activation function here, will use CrossEntropyLoss later
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare(model, layers):
    if layers[0] == 196:
        _, test_images, _ = dnn_datasets()
    elif layers[0] == 784:
        test_images, _, _ = dnn_datasets()

    with torch.no_grad():  # Ensure gradients are not computed
        predictions = model(test_images)
        predicted_labels = predictions.argmax(dim=1)

    predicted_labels = predicted_labels.tolist()
    return predicted_labels, test_images

def prepare_cnn(model, layers):
    if layers[0] == 14:
        _, test_images = cnn_datasets()
    elif layers[0] == 28:
        test_images, _= cnn_datasets()

    with torch.no_grad():  # Ensure gradients are not computed
        predictions = model(test_images)
        predicted_labels = predictions.argmax(dim=1)

    predicted_labels = predicted_labels.tolist()
    return predicted_labels, test_images

def gen_model_dnn(layers, state_dict):
    if len(layers) == 3:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

    elif len(layers) == 4:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])
                self.fc3 = nn.Linear(layers[2], layers[3])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

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
    #parser = argparse.ArgumentParser(description="Generate benchmark result for a given model and testsize.")
    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--model', type=str, help='Model file path')

    # Mutually exclusive for showing models only
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument('--list', action='store_true', help='Show list of supported models and exit')

    parser.add_argument('--save', action='store_true', help='Flag to indicate if save results')
    parser.add_argument('--agg', type=int, help='Set the start for aggregating benchmark results')

    parser.add_argument('--accuracy', action='store_true', help='Flag to indicate if use accuracy mode which may sacrifice efficiency')

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

    start = 0
    notes = ""
    if args.agg:
        start = args.agg
        notes = f'start from {start}'

    if layers[0] > 30:
        dnn = True
    else:
        dnn = False
        
    if dnn:
        arch_folder = "input" + (len(layers)-1) * "-dense" + "/"

        model_path = "../../models/"

        state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

        output_folder = './tmp/' + "_".join([str(x) for x in layers]) + "/"
        os.makedirs(output_folder, exist_ok=True)

        model = gen_model_dnn(layers, state_dict)

        if args.accuracy:
            mode = "accuracy"
        else:
            mode = "resources"

        predicted_labels, tests = prepare(model, layers)
        benchmark_dnn(tests[start:start+args.size], predicted_labels[start:start+args.size], model, args.model, 
                    mode=mode, save=args.save, notes=notes)
    else:
        arch_folder = arch_folders[args.model]
        
        state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

        model = gen_model_cnn(layers, state_dict)
        
        predicted_labels, tests = prepare_cnn(model, layers)

        benchmark_cnn(tests[:args.size], predicted_labels[:args.size], model, args.model, 
                save=args.save)


    #, args.size, test_images_pt, test_labels_pt, args.model, args.save
    
    #benchmark(tests, predicted_labels, program, model_in_path, args.model, args.save)