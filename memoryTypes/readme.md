# CUDA Memory Performance Benchmark

## Overview
This project benchmarks different CUDA memory types by performing a simple **vector addition** operation using:

1. **Global Memory** (Slowest)
2. **Constant Memory** (Cached, faster than global)
3. **Shared Memory** (Fast, on-chip)
4. **Registers** (Fastest, but limited)
5. **Local Memory** (Stored in global memory, slow)

The program measures the **execution time** for each memory type and displays a comparison table.

## Features
- Measures execution time for each CUDA memory type
- Demonstrates performance differences
- Helps optimize CUDA applications by choosing the best memory type

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- GCC or compatible compiler

## Compilation & Execution
### **Step 1: Compile the Program**
```sh
nvcc cuda_memory_benchmark.cu -o memory_benchmark
```

### **Step 2: Run the Benchmark**
```sh
./memory_benchmark
```

## Expected Output
The program outputs a **performance comparison table**, for example:
```
CUDA Memory Performance Comparison
-----------------------------------
| Memory Type  | Execution Time (ms) |
|-------------|--------------------|
| Global      |     5.4321 ms       |
| Constant    |     2.6789 ms       |
| Shared      |     1.3456 ms       |
| Registers   |     0.9876 ms       |
| Local       |     5.7654 ms       |
-----------------------------------
```

## Explanation of Memory Types
### **1. Global Memory**
- Located in **off-chip DRAM**
- **High latency** (~400-600 cycles)
- Should be **minimized** using shared memory

### **2. Constant Memory**
- Read-only memory, **cached**
- **Faster** than global memory for **read-only data**

### **3. Shared Memory**
- **On-chip**, shared within a **thread block**
- **Much faster** than global memory (~100x)
- Reduces redundant global memory accesses

### **4. Registers**
- **Fastest memory**, per-thread
- **Zero latency**, but **limited in size**
- Excessive use can lead to **register spilling**

### **5. Local Memory**
- **Not really local!** Stored in **global memory**
- Used when registers **overflow**
- **Slow**, similar to global memory

## Optimization Strategies
- **Use shared memory** to reduce global memory accesses.
- **Use registers** for frequently accessed values.
- **Avoid local memory usage**, as it’s stored in global memory.
- **Use constant memory** for read-only data shared across threads.

## License
This project is open-source and licensed under the **MIT License**.

