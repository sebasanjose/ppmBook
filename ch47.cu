#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel() {
    int x = threadIdx.x; // Some computation
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxWarpsPerSM = maxThreadsPerSM / 32; // Since 1 warp = 32 threads
    int registersPerSM = prop.regsPerBlock;

    printf("Max Threads Per SM: %d\n", maxThreadsPerSM);
    printf("Max Warps Per SM: %d\n", maxWarpsPerSM);
    printf("Registers Per SM: %d\n", registersPerSM);

    // Check actual kernel register usage
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, myKernel);
    printf("Registers Used Per Thread: %d\n", attr.numRegs);

    return 0;
}
