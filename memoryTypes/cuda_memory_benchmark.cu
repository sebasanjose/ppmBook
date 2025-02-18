#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024 * 1024  // 1 million elements
#define BLOCK_SIZE 256

__constant__ float d_constA[N];  // Constant memory

// Kernel using global memory
__global__ void vectorAddGlobal(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Kernel using constant memory
__global__ void vectorAddConstant(float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = d_constA[i] + B[i];
    }
}

// Kernel using shared memory
__global__ void vectorAddShared(float *A, float *B, float *C) {
    __shared__ float sharedA[BLOCK_SIZE];  // Shared memory buffer

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        sharedA[threadIdx.x] = A[i];  // Load into shared memory
        __syncthreads();  // Ensure all threads have loaded data

        C[i] = sharedA[threadIdx.x] + B[i];
    }
}

// Kernel using registers
__global__ void vectorAddRegisters(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float regA = A[i];  // Load into register
        C[i] = regA + B[i];
    }
}

// Kernel using local memory (actually stored in global memory)
__global__ void vectorAddLocal(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float localA = A[i];  // Local memory (stored in global memory)
        C[i] = localA + B[i];
    }
}

// Timer function
float measureTime(void (*kernel)(float*, float*, float*), float *A, float *B, float *C, float *d_A, float *d_B, float *d_C) {
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float measureTimeConstant(void (*kernel)(float*, float*), float *B, float *C, float *d_B, float *d_C) {
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_B, d_C);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}


int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data to constant memory
    cudaMemcpyToSymbol(d_constA, h_A, N * sizeof(float));

    // Measure times
    float timeGlobal = measureTime(vectorAddGlobal, h_A, h_B, h_C, d_A, d_B, d_C);
    float timeConstant = measureTimeConstant(vectorAddConstant, h_B, h_C, d_A, d_B, d_C);
    float timeShared = measureTime(vectorAddShared, h_A, h_B, h_C, d_A, d_B, d_C);
    float timeRegisters = measureTime(vectorAddRegisters, h_A, h_B, h_C, d_A, d_B, d_C);
    float timeLocal = measureTime(vectorAddLocal, h_A, h_B, h_C, d_A, d_B, d_C);

    // Print results
    printf("\nCUDA Memory Performance Comparison\n");
    printf("-----------------------------------\n");
    printf("| Memory Type  | Execution Time (ms) |\n");
    printf("|-------------|--------------------|\n");
    printf("| Global      | %10.4f ms       |\n", timeGlobal);
    printf("| Constant    | %10.4f ms       |\n", timeConstant);
    printf("| Shared      | %10.4f ms       |\n", timeShared);
    printf("| Registers   | %10.4f ms       |\n", timeRegisters);
    printf("| Local       | %10.4f ms       |\n", timeLocal);
    printf("-----------------------------------\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
