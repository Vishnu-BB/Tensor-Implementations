import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import struct
import os
import time

# --- Configuration ---
SEED = 42
BATCH_SIZE = 64
INPUT_DIM = 128
HIDDEN_1 = 256
HIDDEN_2 = 128
OUTPUT_DIM = 10
LEARNING_RATE = 0.01
EPOCHS = 5
DATA_FILE = "accuracy_test_data.bin"
RESULTS_FILE = "accuracy_test_results.bin"
EXECUTABLE = "../build/Tests/test_accuracy" 
# Note: Executable path might change depending on how we run it (make run-snippet vs verify_accuracy target)
# We will assume we compile it to a specific location or pass it as an argument.
# For now, let's assume we run via `make run-snippet FILE=Tests/test_accuracy.cpp` and capture output, 
# BUT `make run-snippet` cleans up the binary. 
# Better: We will rely on the C++ test to read/write files and this script just orchestrates.

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(HIDDEN_2, OUTPUT_DIM)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

def save_tensor(tensor, f):
    # Save dims count
    dims = list(tensor.shape)
    f.write(struct.pack('I', len(dims)))
    # Save dims
    for d in dims:
        f.write(struct.pack('I', d))
    # Save data (float32)
    data = tensor.detach().numpy().astype(np.float32)
    data.tofile(f)

def run_pytorch_verification():
    """Runs one iteration to get ground truth values for accuracy check."""
    set_seed(SEED)
    input_data = torch.randn(BATCH_SIZE, INPUT_DIM)
    target_data = torch.randn(BATCH_SIZE, OUTPUT_DIM)
    model = TestModel()
    
    # Save for C++
    with open(DATA_FILE, 'wb') as f:
        save_tensor(input_data, f)
        save_tensor(target_data, f)
        save_tensor(model.fc1.weight, f)
        save_tensor(model.fc1.bias, f)
        save_tensor(model.fc2.weight, f)
        save_tensor(model.fc2.bias, f)
        save_tensor(model.fc3.weight, f)
        save_tensor(model.fc3.bias, f)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    optimizer.zero_grad()
    x1 = model.fc1(input_data)
    relu_out = model.relu1(x1)
    x2 = model.fc2(relu_out)
    sigmoid_out = model.sigmoid(x2)
    final_out = model.fc3(sigmoid_out)
    loss = criterion(final_out, target_data)
    loss.backward()
    optimizer.step()
    
    return {
        "relu": relu_out,
        "sigmoid": sigmoid_out,
        "final": final_out,
        "loss": loss,
        "fc1_weight_new": model.fc1.weight.clone()
    }

def benchmark_pytorch(device_str, iterations=100):
    """Measures avg time per iteration for PyTorch on a specific device."""
    device = torch.device("cuda" if device_str.lower() == "cuda" else "cpu")
    
    # Check if CUDA is available for PyTorch
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[Python] CUDA requested for PyTorch but not available. Falling back to CPU.")
        device = torch.device("cpu")

    model = TestModel().to(device)
    input_data = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    target_data = torch.randn(BATCH_SIZE, OUTPUT_DIM).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(input_data)
        l = criterion(out, target_data)
        l.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        optimizer.zero_grad()
        out = model(input_data)
        l = criterion(out, target_data)
        l.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    avg_ms = ((time.time() - start_time) * 1000) / iterations
    return avg_ms

def read_cpp_results():
    if not os.path.exists(RESULTS_FILE):
        return None
        
    with open(RESULTS_FILE, 'rb') as f:
        # Helper to read tensor
        def read_tensor(f):
            dims_len = struct.unpack('I', f.read(4))[0]
            dims = []
            for _ in range(dims_len):
                dims.append(struct.unpack('I', f.read(4))[0])
            
            num_elements = 1
            for d in dims:
                num_elements *= d
                
            data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
            return data.reshape(dims)
            
        res = {}
        try:
            res["relu"] = read_tensor(f)
            res["sigmoid"] = read_tensor(f)
            res["final"] = read_tensor(f)
            res["loss"] = read_tensor(f) # Scalar usually 1x1 
            res["fc1_grad"] = read_tensor(f)
            res["fc1_weight_new"] = read_tensor(f)
            res["throughput"] = struct.unpack('f', f.read(4))[0]
        except Exception as e:
            print(f"Error reading C++ results: {e}")
            return None
            
    return res

def compare_and_print(py_res, cpp_res, dev_name="OwnTensor"):
    if cpp_res is None:
        print(f"Failed to load C++ results for {dev_name}.")
        return

    # Helper to print format
    def print_item(name, py_val, cpp_val):
        # py_val is Tensor, cpp_val is numpy array
        if isinstance(py_val, torch.Tensor):
            p_v = py_val.detach().numpy()
        else:
            p_v = py_val
            
        c_v = cpp_res[name]
        
        # Mean value for display
        p_mean = np.mean(p_v)
        c_mean = np.mean(c_v)
        
        # Diff
        diff = np.abs(p_v - c_v).max()
        
        # Accuracy check
        # User requested: "In OwnTensor Relu: value:0.009 |Accuracy:100%"
        # We define Accuracy 100% if diff < epsilon
        acc = "100%" if diff < 1e-4 else "0%"
        
        print(f"In OwnTensor({dev_name}) {name}: value:{c_mean:.7f} | Accuracy:{acc}")
        print(f"In Pytorch {name}: value:{p_mean:.7f} | Accuracy:100%")
        print(f"Diff in {name}(btw pytorch and Owntensor value)={diff:.7f}")
        print("............")

    print_item("relu", py_res["relu"], cpp_res) 
    print_item("sigmoid", py_res["sigmoid"], cpp_res)
    print_item("final", py_res["final"], cpp_res) 
    print_item("loss", py_res["loss"], cpp_res)
    print_item("fc1_weight_new", py_res["fc1_weight_new"], cpp_res)
    
    print(f"Throughput PyTorch: {py_res['throughput']:.2f} ms")
    print(f"Throughput OwnTensor({dev_name}): {cpp_res['throughput']:.2f} ms")

def compare_and_print_refined(py_res, cpp_res, dev_name):
    # Helper to print format
    def print_item(name, py_val, cpp_val):
        if isinstance(py_val, torch.Tensor):
            p_v = py_val.detach().cpu().numpy()
        else:
            p_v = py_val
        c_v = cpp_res[name]
        p_mean = np.mean(p_v)
        c_mean = np.mean(c_v)
        diff = np.abs(p_v - c_v).max()
        acc = "100%" if diff < 1e-4 else "0%"
        print(f"In OwnTensor({dev_name}) {name}: value:{c_mean:.7f} | Accuracy:{acc}")
        print(f"In Pytorch({dev_name}) {name}: value:{p_mean:.7f} | Accuracy:100%")
        print(f"Diff in {name}(btw pytorch and Owntensor value)={diff:.7f}")
        print("............")

    print_item("relu", py_res["relu"], cpp_res) 
    print_item("sigmoid", py_res["sigmoid"], cpp_res)
    print_item("final", py_res["final"], cpp_res) 
    print_item("loss", py_res["loss"], cpp_res)
    print_item("fc1_weight_new", py_res["fc1_weight_new"], cpp_res)
    
    print(f"Throughput PyTorch({dev_name}): {py_res['py_throughput']:.2f} ms")
    print(f"Throughput OwnTensor({dev_name}): {py_res['own_throughput']:.2f} ms")

if __name__ == "__main__":
    print("[Python] Starting Accuracy and  Throughput Benchmark...")
    py_verif = run_pytorch_verification()
    
    print("[Python] Building C++...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpp_file = "Tests/test_accuracy.cpp"
    
    devices = ["CPU", "CUDA"]
    
    for dev in devices:
        print(f"\n==================================================")
        print(f"Testing Device: {dev}")
        print(f"==================================================")
        
        # 1. Run OwnTensor
        env = os.environ.copy()
        env["TEST_DEVICE"] = dev
        ldlibs = "-lcudart -ltbb -lcurand -lcublas"
        cxxflags = "-std=c++20 -fPIC -Wall -Wextra -g -fopenmp -O3 -march=native"
        cmd = ["make", "run-snippet", f"FILE={cpp_file}", f"LDLIBS={ldlibs}", f"CXXFLAGS={cxxflags}"]
        
        process = subprocess.Popen(cmd, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"C++ Execution Failed for {dev}:")
            print(stdout.decode())
            print(stderr.decode())
            continue
        
        cpp_res = read_cpp_results()
        if cpp_res is None:
            print(f"Failed to read OwnTensor results for {dev}")
            continue

        # 2. Run matched PyTorch Benchmark
        print(f"[Python] Running matched PyTorch benchmark on {dev}...")
        py_throughput = benchmark_pytorch(dev)
        
        # 3. Combine and Compare
        results_combined = py_verif.copy()
        results_combined["py_throughput"] = py_throughput
        results_combined["own_throughput"] = cpp_res["throughput"]
        
        compare_and_print_refined(results_combined, cpp_res, dev)
        
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)

