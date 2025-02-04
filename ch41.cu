#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to add two arrays in parallel
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int size = 1024;  // Size of arrays
    int bytes = size * sizeof(int);

    // Allocate host memory
    int *h_a = new int[size];
    int *h_b = new int[size];
    int *h_c = new int[size];

    // Initialize arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;   // Number of threads per block
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print some results
    for (int i = 0; i < 10; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // Free memory
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
