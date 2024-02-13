import torch, struct, os, psutil, subprocess, time, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from torch.utils.data import DataLoader, TensorDataset
import concurrent.futures
import pandas as pd
import argparse

params = {"784_56_10": 44543,
          "196_25_10": 5185,
          "196_24_14_10": 5228}

accuracys = {"784_56_10": 0.9740,
          "196_25_10": 0.9541,
          "196_24_14_10": 0.9556}

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

def load_img_from_file(data_file="input", show=False):
    try:
        with open(data_file, 'rb') as file:
            buf = file.read()
    except Exception as e:
        print(e)
        return None, e

    digits = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    l = int(np.sqrt(len(digits)))
    if show:
        c = ""
        for row in range(l):
            for col in range(l):
                if buf[row * l + col] > 230:
                    c += "&"
                else:
                    c += "-"
            c += "\n"
        print(c)

    return digits, None

def save_img_to_file(image, data_file = "input"):
    try:
        # Convert to bytes
        image_bytes = np.array(image*255).astype('uint8').tobytes()
        
        # Write to file
        with open(data_file, 'wb') as file:
            file.write(image_bytes)
        
        #print(f"Image saved to {data_file}")
    except Exception as e:
        print(f"Error saving image: {e}")

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

def calculate_loss(veri_infer, predicted_labels):
    count = 0
    for i in range(len(veri_infer)):
        if veri_infer[i] != predicted_labels[i]:
            count +=1
            print (f"Index {i} Not match!")

    return count/len(veri_infer)*100

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

def gen_model(layers, state_dict):
    if len(layers) == 3:
        class Net(nn.Module):
            def __init__(self, num_classes=10):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(layers[0], layers[1])  # Flatten 
                self.fc2 = nn.Linear(layers[1], layers[2])  

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.softmax(x, dim = 1)

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
                return F.softmax(x, dim = 1)
    else:
        print ("Layers not Support")
        return None
    
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare(model, model_name, state_dict, test_images):
    with torch.no_grad():  # Ensure gradients are not computed
        predictions = model(test_images)
        predicted_labels = predictions.argmax(dim=1)

    predicted_labels = predicted_labels.tolist()

    folder = "./tmp/"
    # Create the directory 'tmp' in the current working directory
    os.makedirs(folder, exist_ok=True)

    fname_out = "./bin/" + arch_folder + "ggml-model-" + model_name + ".bin"
    pack_fmt = "!i"

    os.makedirs("./bin/" + arch_folder, exist_ok=True)

    fout = open(fname_out, "w+b")
    fout.write(struct.pack(pack_fmt, 0x67676d6c)) # magic: ggml in hex

    for name in state_dict.keys():
        data = state_dict[name].squeeze().numpy()
        #print("Processing variable: " + name + " with shape: ", data.shape) 
        n_dims = len(data.shape)
    
        fout.write(struct.pack(pack_fmt, n_dims))
        
        data = data.astype(np.float32)
        for i in range(n_dims):
            fout.write(struct.pack(pack_fmt, data.shape[n_dims - 1 - i]))

        # data
        data = data.astype(">f4")
        data.tofile(fout)

    fout.close()

    return predicted_labels, fname_out

def benchmark(test_images, predicted_labels, program, 
              model_in_path, model_name, save = False):
    vm_file = "./opml/mlvm/mlvm"
    if not os.path.exists(vm_file):
        # File does not exist, print message and return from the function
        print("Need to run the setup, by calling bash code 'bash ./setup-opml.sh'")
        return
    
    # If the file exists, proceed with the rest of the function
    print("File exists, proceeding with the operation.")

    #program = "./bin/mlgo_196_IDD.bin"
    tmp_folder = './tmp/'

    benchmark_start_time = time.time()
    loss = 0
    mem_usage = []
    time_cost = []
    for ind, img in enumerate(test_images):
        img_out_path = tmp_folder + str(ind)
        save_img_to_file(img, img_out_path)

        # Exclusion of Pre-processing
        start_time = time.time()
        command = [f"{vm_file}", f"--basedir={tmp_folder}",
                f"--program={program}", f"--model={model_in_path}", 
                f"--data={img_out_path}", "--mipsVMCompatible"]
        
        print ("Process for image", ind)
        # subprocess.run(command)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get the process ID
        pid = process.pid
        # print ("pid:", pid)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(monitor_memory, pid)
            _, stderr = process.communicate()
            max_memory = future.result()

        try:
            pred = int(stderr[-2])
        except ValueError:
            print(f"Failed to convert {stderr[-2]} to int. Full output: {stderr}")
            pred  = -1

        if pred != predicted_labels[ind]:
            print ("Loss on index", ind, pred, predicted_labels[ind])
            loss += 1

        mem_usage.append(max_memory)
        time_cost.append(time.time() - start_time)
    
    print ("Total time:", time.time() - benchmark_start_time)

    layers = model_name.split("_")
    new_row = {
        'Framework': ['opml (pytorch)'],
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

def show_models():
    for key in params:
        layers = key.split("_")
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

    args = parser.parse_args()

    if args.list:
        show_models()
        sys.exit()

    if args.model not in params:
        print ("Please check the model name by using '--list'")
        sys.exit()

    if not args.model or args.size is None:
        parser.error('--model and --size are required for benchmarking.')


    # Load TensorFlow MNIST data
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
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

    model_path = "../../models/"
    layers = [int(x) for x in args.model.split("_")]
    arch_folder = "input" + (len(layers)-1) * "-dense" + "/"

    state_dict = torch.load(model_path + arch_folder+ args.model + ".pth")

    model = gen_model(layers, state_dict)

    if not args.save:
        args.save = False

    if layers[0] == 784 and len(layers) == 3:
        program = "./bin/mlgo_784_IDD.bin"
        predicted_labels, model_in_path = prepare(model, args.model, state_dict, test_images_pt)
        tests = test_images_pt[:args.size]
    elif layers[0] == 196 and len(layers) == 3:
        program = "./bin/mlgo_196_IDD.bin"
        predicted_labels, model_in_path = prepare(model, args.model, state_dict, test_images_pt_downsampled)
        tests = test_images_pt_downsampled[:args.size]
    elif layers[0] == 196 and len(layers) == 4:
        program = "./bin/mlgo_196_IDDD.bin"
        predicted_labels, model_in_path = prepare(model, args.model, state_dict, test_images_pt_downsampled)
        tests = test_images_pt_downsampled[:args.size]        
    else:
        print ("format not support")
        sys.exit()

    #, args.size, test_images_pt, test_labels_pt, args.model, args.save
    
    benchmark(tests, predicted_labels, program, model_in_path, args.model, args.save)
