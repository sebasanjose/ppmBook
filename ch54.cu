#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Tile size for shared memory optimization

// Naïve matrix multiplication (No tiling, inefficient)
__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // Multiple global memory accesses
        }
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication using tiling
__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0;

    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            A_shared[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            B_shared[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();  // Synchronize threads before computing

        // Compute partial result
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();  // Ensure shared memory is not overwritten before next iteration
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Helper function to initialize matrices
void initializeMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

// Function to compare results
bool verifyResult(float *C1, float *C2, int N) {
    for (int i = 0; i < N * N; i++) {
        if (abs(C1[i] - C2[i]) > 1e-4) {
            printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int N = 512;  // Matrix size N x N
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_tiled = (float *)malloc(size);

    // Initialize matrices
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_naive, size);
    cudaMalloc(&d_C_tiled, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Measure time for Naïve Kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);

    // Measure time for Optimized Tiled Kernel
    cudaEventRecord(start);
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiledTime = 0;
    cudaEventElapsedTime(&tiledTime, start, stop);

    cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResult(h_C_naive, h_C_tiled, N)) {
        printf("Results match! Optimized kernel is correct.\n");
    } else {
        printf("Mismatch found! There is an error in the computation.\n");
    }

    // Print performance comparison
    printf("\nMatrix Multiplication Performance Comparison\n");
    printf("--------------------------------------------\n");
    printf("| Method       | Execution Time (ms) |\n");
    printf("|-------------|--------------------|\n");
    printf("| Naïve       | %10.4f ms        |\n", naiveTime);
    printf("| Tiled       | %10.4f ms        |\n", tiledTime);
    printf("--------------------------------------------\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);

    return 0;
}
