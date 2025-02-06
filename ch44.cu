#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64  // 64 threads (2 warps)
#define WARP_SIZE 32   // 32 threads per warp

// 1️⃣ Warp Info: Print each thread’s warp ID
__global__ void warpInfo() {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;  // Compute warp ID

    printf("[Warp Info] Thread %d belongs to Warp %d in Block %d\n", tid, warpId, blockIdx.x);
}

// 2️⃣ Warp Divergence Example: Different execution paths cause serialization
__global__ void warpDivergence() {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;

    if (tid % 2 == 0) {
        printf("[Divergence] Thread %d (Warp %d) takes Path A\n", tid, warpId);
    } else {
        printf("[Divergence] Thread %d (Warp %d) takes Path B\n", tid, warpId);
    }
}

// 3️⃣ Optimized Warp Execution: Uses predication to avoid divergence
__global__ void optimizedWarpExecution() {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;

    // Uses predication instead of branching
    int value = (tid % 2 == 0) ? 10 : 20;
    
    printf("[Optimized] Thread %d (Warp %d) computes value: %d\n", tid, warpId, value);
}

// 4️⃣ Warp Shuffle Reduction: Fast SIMD-style reduction
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void warpShuffleReduction(int *input, int *output) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;  // Thread position within warp
    int warpId = tid / WARP_SIZE;

    int sum = input[tid];

    // Perform warp-wide reduction using registers
    sum = warpReduceSum(sum);

    // Store the result for each warp
    if (lane == 0) output[warpId] = sum;
}

// Function to run a CUDA kernel and print execution time
void runKernel(void (*kernel)(), const char *label) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<1, BLOCK_SIZE>>>(); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%s executed in %.6f ms\n", label, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Run each kernel and measure execution time
    printf("\n=== Running Warp Execution Comparisons ===\n");

    runKernel(warpInfo, "Warp Information");
    runKernel(warpDivergence, "Warp Divergence Example");
    runKernel(optimizedWarpExecution, "Optimized Warp Execution");

    // Prepare input for warp shuffle reduction
    int h_input[BLOCK_SIZE], h_output[BLOCK_SIZE / WARP_SIZE];
    int *d_input, *d_output;
    cudaMalloc(&d_input, BLOCK_SIZE * sizeof(int));
    cudaMalloc(&d_output, (BLOCK_SIZE / WARP_SIZE) * sizeof(int));

    for (int i = 0; i < BLOCK_SIZE; i++) h_input[i] = 1; // Set all elements to 1

    cudaMemcpy(d_input, h_input, BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    warpShuffleReduction<<<1, BLOCK_SIZE>>>(d_input, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Warp Shuffle Reduction executed in %.6f ms\n", milliseconds);

    cudaMemcpy(h_output, d_output, (BLOCK_SIZE / WARP_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    printf("[Reduction] Warp 0 Sum: %d (Expected: 32)\n", h_output[0]);
    printf("[Reduction] Warp 1 Sum: %d (Expected: 32)\n", h_output[1]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
