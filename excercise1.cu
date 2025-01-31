#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 512 // Define matrix size (WIDTH x WIDTH)

// Row-wise kernel: Each thread computes a row of P
__global__ void MatrixMulRowKernel(float *M, float *N, float *P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles a row

    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// Column-wise kernel: Each thread computes a column of P
__global__ void MatrixMulColumnKernel(float *M, float *N, float *P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles a column

    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// Function to initialize matrices with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 10; // Random numbers between 0 and 9
    }
}

// Function to measure execution time
float measureExecutionTime(void (*kernel)(float*, float*, float*, int), float* d_M, float* d_N, float* d_P, dim3 gridDim, dim3 blockDim, int Width) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, Width);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    int size = WIDTH * WIDTH * sizeof(float);

    // Allocate host memory
    float *hostM = (float*)malloc(size);
    float *hostN = (float*)malloc(size);
    float *hostP = (float*)malloc(size);

    // Initialize matrices with random values
    initializeMatrix(hostM, WIDTH * WIDTH);
    initializeMatrix(hostN, WIDTH * WIDTH);

    // Allocate device memory
    float *deviceM, *deviceN, *deviceP;
    cudaMalloc((void **)&deviceM, size);
    cudaMalloc((void **)&deviceN, size);
    cudaMalloc((void **)&deviceP, size);

    // Copy matrices M and N to device
    cudaMemcpy(deviceM, hostM, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, hostN, size, cudaMemcpyHostToDevice);

    // Define execution configurations
    dim3 blockDim(16); // 16 threads per block
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x); // Enough blocks to cover all rows/columns

    // Measure execution time for row-wise kernel
    float rowTime = measureExecutionTime(MatrixMulRowKernel, deviceM, deviceN, deviceP, gridDim, blockDim, WIDTH);

    // Measure execution time for column-wise kernel
    float columnTime = measureExecutionTime(MatrixMulColumnKernel, deviceM, deviceN, deviceP, gridDim, blockDim, WIDTH);

    // Print results
    printf("Row-wise Kernel Time: %.3f ms\n", rowTime);
    printf("Column-wise Kernel Time: %.3f ms\n", columnTime);

    // Free memory
    cudaFree(deviceM);
    cudaFree(deviceN);
    cudaFree(deviceP);
    free(hostM);
    free(hostN);
    free(hostP);

    return 0;
}
