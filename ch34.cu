#include <stdio.h>
#include <cuda_runtime.h>

// Define matrix size
#define WIDTH 4 

// CUDA Kernel for basic matrix multiplication
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width) {
    // Calculate row and column indices for the element this thread will compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we are within matrix bounds
    if (row < Width && col < Width) {
        float Pvalue = 0;

        // Compute the dot product of row M[row] and column N[col]
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }

        // Store the result in matrix P
        P[row * Width + col] = Pvalue;
    }
}

int main() {
    int size = WIDTH * WIDTH * sizeof(float);

    // Allocate memory for matrices on host (CPU)
    float hostM[WIDTH * WIDTH] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    printf("Matrix M:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.2f ", hostM[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("Matrix N:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.2f ", hostN[i * WIDTH + j]);
        }
        printf("\n");
    }

    float hostN[WIDTH * WIDTH] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float hostP[WIDTH * WIDTH] = {0}; // Output matrix

    // Allocate memory on device (GPU)
    float *deviceM, *deviceN, *deviceP;
    cudaMalloc((void **)&deviceM, size);
    cudaMalloc((void **)&deviceN, size);
    cudaMalloc((void **)&deviceP, size);

    // Copy matrices M and N to device
    cudaMemcpy(deviceM, hostM, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, hostN, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(2, 2); // Each block contains 2x2 threads
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (WIDTH + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    MatrixMulKernel<<<gridDim, blockDim>>>(deviceM, deviceN, deviceP, WIDTH);

    // Copy result back to host
    cudaMemcpy(hostP, deviceP, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    printf("Matrix P (Result of M x N):\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.2f ", hostP[i * WIDTH + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(deviceM);
    cudaFree(deviceN);
    cudaFree(deviceP);

    return 0;
}
