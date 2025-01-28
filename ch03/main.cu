// matrix multiplication
#include <stdio.h>

// Kernel function to square each element
__global__ void squareMatrix(int *matrix, int width, int height) {
    // Calculate the thread's global x and y index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    // Check if the thread is within bounds
    if (row < height && col < width) {
        int idx = row * width + col; // Convert 2D index to 1D
        matrix[idx] = matrix[idx] * matrix[idx]; // Square the element
    }
}

int main() {
    // Define 2D array dimensions
    const int width = 6;
    const int height = 4;

    // Allocate and initialize host matrix
    int hostMatrix[width * height];
    for (int i = 0; i < width * height; i++) {
        hostMatrix[i] = i + 1; // Fill with numbers 1 to width*height
    }

        // Print the squared matrix
    printf("Original Matrix:\n");
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%d ", hostMatrix[row * width + col]);
        }
        printf("\n");
    }

    // Allocate device memory
    int *deviceMatrix;
    cudaMalloc((void **)&deviceMatrix, width * height * sizeof(int));

    // Copy matrix to device
    cudaMemcpy(deviceMatrix, hostMatrix, width * height * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(2, 2);               // 2x2 threads per block
    dim3 gridDim((width + 1) / 2, (height + 1) / 2); // Enough blocks to cover matrix

    // Launch the kernel
    squareMatrix<<<gridDim, blockDim>>>(deviceMatrix, width, height);

    // Copy result back to host
    cudaMemcpy(hostMatrix, deviceMatrix, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the squared matrix
    printf("Squared Matrix:\n");
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%d ", hostMatrix[row * width + col]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(deviceMatrix);

    return 0;
}
