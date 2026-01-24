import torch
import torch.nn as nn
import torch.optim as optim
import os
import psutil
import struct

# --- Configuration (Matches test_memory.cpp) ---
BATCH_SIZE = 64
INPUT_DIM = 128
HIDDEN_1 = 256
HIDDEN_2 = 128
OUTPUT_DIM = 10
LEARNING_RATE = 0.01

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

def benchmark(device_str):
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

    for _ in range(100):
        optimizer.zero_grad()
        out = model(input_data)
        l = criterion(out, target_data)
        l.backward()
        optimizer.step()
        
        # Track peak RSS
        peak_cpu = max(peak_cpu, psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
        if device.type == "cuda":
            # For fair comparison with OwnTensor (which might include context/reserved), 
            # we use memory_reserved which includes the caching pool.
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_reserved(device) / (1024 * 1024))
            
    # Save to a temporary file for the runner to read
    with open("py_memory_temp.bin", "wb") as f:
        f.write(struct.pack('f', peak_cpu))
        f.write(struct.pack('f', peak_cuda))

if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "CPU"
    benchmark(dev)
