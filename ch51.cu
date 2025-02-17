#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Define tile size for shared memory optimization

// Naïve matrix multiplication kernel
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

// Optimized matrix multiplication using shared memory
__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            A_shared[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            B_shared[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();  // Synchronize to ensure all threads have loaded data

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();  // Ensure shared memory is not overwritten before next loop
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
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Run Naïve Kernel
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, N);
    cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);

    // Run Optimized Kernel
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, N);
    cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResult(h_C_naive, h_C_tiled, N)) {
        printf("Results match! Optimized kernel is correct.\n");
    } else {
        printf("Mismatch found! There is an error in the computation.\n");
    }

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
