#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Tile size for shared memory optimization

// Naïve matrix multiplication (No tiling, inefficient)
__global__ void matMulNaive(float *A, float *B, float *C, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0.0;
        for (int k = 0; k < Width; k++) {
            Pvalue += A[Row * Width + k] * B[k * Width + Col];  // Multiple global memory accesses
        }
        C[Row * Width + Col] = Pvalue;
    }
}

// Optimized matrix multiplication using tiling with boundary checks
__global__ void matMulTiled(float *A, float *B, float *C, int Width) {
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // Load tiles into shared memory with boundary checks
        if (Row < Width && (ph * TILE_WIDTH + threadIdx.x) < Width)
            A_shared[threadIdx.y][threadIdx.x] = A[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;  // Prevents using invalid memory

        if ((ph * TILE_WIDTH + threadIdx.y) < Width && Col < Width)
            B_shared[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;  // Prevents using invalid memory

        __syncthreads();  // Ensure all threads have loaded their values

        // Compute partial result
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();  // Ensure shared memory is not overwritten before next iteration
    }

    // Store the final result with boundary check
    if (Row < Width && Col < Width)
        C[Row * Width + Col] = Pvalue;
}

// Function to initialize matrices with random values
void initializeMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width * Width; i++) {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

// Function to compare results
bool verifyResult(float *C1, float *C2, int Width) {
    for (int i = 0; i < Width * Width; i++) {
        if (abs(C1[i] - C2[i]) > 1e-4) {
            printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int Width = 2048;  // Large matrix size
    size_t size = Width * Width * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_tiled = (float *)malloc(size);

    // Initialize matrices
    initializeMatrix(h_A, Width);
    initializeMatrix(h_B, Width);

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
    dim3 gridSize((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Measure time for Naïve Kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);

    // Measure time for Optimized Tiled Kernel
    cudaEventRecord(start);
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiledTime = 0;
    cudaEventElapsedTime(&tiledTime, start, stop);

    cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResult(h_C_naive, h_C_tiled, Width)) {
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
