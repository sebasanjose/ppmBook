#include <iostream>
#include <cuda_runtime.h>

__global__ void blockSchedulingExample() {
    int blockId = blockIdx.x;
    printf("Executing Block %d on an SM\n", blockId);
}

int main() {
    int numBlocks = 8;      // Total blocks to schedule
    int threadsPerBlock = 1; // One thread per block for clarity

    // Launch the kernel with multiple blocks
    blockSchedulingExample<<<numBlocks, threadsPerBlock>>>();

    // Synchronize GPU and CPU to ensure all prints are captured
    cudaDeviceSynchronize();

    return 0;
}
