#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        printf("Device %d: %s\n", i, devProp.name);
        printf("  - Compute Capability: %d.%d\n", devProp.major, devProp.minor);
        printf("  - Multiprocessors: %d\n", devProp.multiProcessorCount);
        printf("  - Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);
        printf("  - Warp Size: %d\n", devProp.warpSize);
        printf("  - Max Grid Size: (%d, %d, %d)\n",
               devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("  - Max Threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("  - Registers per Block: %d\n", devProp.regsPerBlock);
        printf("  - Shared Memory per Block: %lu bytes\n\n", devProp.sharedMemPerBlock);
    }

    return 0;
}
