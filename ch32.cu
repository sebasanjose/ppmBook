#include <stdio.h>

#define WIDTH 6
#define HEIGHT 6

// CUDA Kernel: Doubles each matrix element
__global__ void scaleMatrix(int *matrix, int width, int height) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread is within valid matrix bounds
    if (row < height && col < width) {
        int idx = row * width + col; // Convert (row, col) to 1D index
        matrix[idx] *= 2;  // Scale value by 2
    }
}

int main() {
    // Allocate and initialize host matrix
    int hostMatrix[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        hostMatrix[i] = i + 1;
    }

    // Allocate device memory
    int *deviceMatrix;
    cudaMalloc((void **)&deviceMatrix, WIDTH * HEIGHT * sizeof(int));

    // Copy matrix to GPU
    cudaMemcpy(deviceMatrix, hostMatrix, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    // Define 2D grid and 2D block dimensions
    dim3 blockDim(2, 2);              // Each block has 2×2 = 4 threads
    dim3 gridDim((WIDTH + 1) / 2, (HEIGHT + 1) / 2); // Enough blocks to cover matrix

    // Launch kernel
    scaleMatrix<<<gridDim, blockDim>>>(deviceMatrix, WIDTH, HEIGHT);

    // Copy result back to host
    cudaMemcpy(hostMatrix, deviceMatrix, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Scaled Matrix:\n");
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            printf("%d ", hostMatrix[row * WIDTH + col]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(deviceMatrix);

    return 0;
}
