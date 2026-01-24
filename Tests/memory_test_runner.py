import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import struct
import os
import time
import psutil

# --- Configuration (Same as Accuracy Test) ---
SEED = 42
BATCH_SIZE = 64
INPUT_DIM = 128
HIDDEN_1 = 256
HIDDEN_2 = 128
OUTPUT_DIM = 10
LEARNING_RATE = 0.01
DATA_FILE = "accuracy_test_data.bin"
RESULTS_FILE = "memory_test_results.bin"

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

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def benchmark_pytorch_memory(device_str, iterations=10):
    device = torch.device("cuda" if device_str.lower() == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = TestModel().to(device)
    input_data = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    target_data = torch.randn(BATCH_SIZE, OUTPUT_DIM).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    peak_cpu = 0
    peak_cuda = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iterations):
        optimizer.zero_grad()
        out = model(input_data)
        l = criterion(out, target_data)
        l.backward()
        optimizer.step()
        
        peak_cpu = max(peak_cpu, get_cpu_memory())
        if device.type == "cuda":
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            
    return peak_cpu, peak_cuda

def read_cpp_memory_results():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, 'rb') as f:
        cpu_mb = struct.unpack('f', f.read(4))[0]
        cuda_mb = struct.unpack('f', f.read(4))[0]
    return cpu_mb, cuda_mb

def get_gpu_memory(pid):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) == 2:
                p, mem = parts
                if int(p.strip()) == pid:
                    return float(mem.strip())
    except Exception:
        pass
    return 0.0

def run_bench(cmd, env, name):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    
    peak_rss = 0
    peak_gpu = 0
    
    try:
        p_parent = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                # Track parent and all children
                procs = [p_parent] + p_parent.children(recursive=True)
                current_rss = 0
                current_gpu = 0
                
                for proc in procs:
                    try:
                        current_rss += proc.memory_info().rss / (1024 * 1024)
                        current_gpu += get_gpu_memory(proc.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                peak_rss = max(peak_rss, current_rss)
                peak_gpu = max(peak_gpu, current_gpu)
                
                time.sleep(0.01) # Faster sampling
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    except Exception:
        pass
        
    stdout, stderr = process.communicate()
    return process.returncode, peak_rss, peak_gpu, stdout.decode(), stderr.decode()

if __name__ == "__main__":
    print("[Python] Starting Memory Benchmark...")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpp_file = "Tests/test_memory.cpp"
    py_bench_script = "Tests/pytorch_mem_benchmark.py"
    devices = ["CPU", "CUDA"]
    
    for dev in devices:
        print(f"\n==================================================")
        print(f"Testing Memory on Device: {dev}")
        print(f"==================================================")
        
        # 1. Run OwnTensor
        env_own = os.environ.copy()
        env_own["TEST_DEVICE"] = dev
        ldlibs = "-lcudart -ltbb -lcurand -lcublas"
        cxxflags = "-std=c++20 -fPIC -Wall -Wextra -g -fopenmp -O3 -march=native"
        cmd_own = ["make", "run-snippet", f"FILE={cpp_file}", f"LDLIBS={ldlibs}", f"CXXFLAGS={cxxflags}"]
        
        ret_own, own_rss, own_gpu, out_own, err_own = run_bench(cmd_own, env_own, "OwnTensor")
        if ret_own != 0:
            print(f"OwnTensor Failed for {dev}.\nSTDOUT: {out_own}\nSTDERR: {err_own}")
            continue

        # 2. Run PyTorch
        cmd_py = ["python3", py_bench_script, dev]
        ret_py, py_rss, py_gpu, out_py, err_py = run_bench(cmd_py, os.environ, "PyTorch")
        if ret_py != 0:
            print(f"PyTorch Failed for {dev}.\nSTDOUT: {out_py}\nSTDERR: {err_py}")
            continue
        
        # 3. Report
        if dev == "CPU":
            print(f"In OwnTensor({dev}) Peak Memory: value:{own_rss:.7f} MB")
            print(f"In Pytorch({dev}) Peak Memory: value:{py_rss:.7f} MB")
            diff = abs(own_rss - py_rss)
            print(f"Diff in Memory(btw pytorch and Owntensor value)={diff:.7f} MB")
        else:
            # For CUDA, use the max of (SMI measurement) or (Internal if we wanted, but SMI is fairer)
            # Actually SMI might miss some peaks if the process is fast.
            # But let's use SMI as the primary "fair" process-level metric.
            print(f"In OwnTensor({dev}) Peak VRAM: value:{own_gpu:.7f} MB")
            print(f"In Pytorch({dev}) Peak VRAM: value:{py_gpu:.7f} MB")
            diff = abs(own_gpu - py_gpu)
            print(f"Diff in VRAM(btw pytorch and Owntensor value)={diff:.7f} MB")
        
        print("............")
