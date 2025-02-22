#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Tile size for shared memory optimization

// Kernel using fixed shared memory (compile-time)
__global__ void matMulFixed(float *A, float *B, float *C, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < Width && (ph * TILE_WIDTH + threadIdx.x) < Width)
            Mds[threadIdx.y][threadIdx.x] = A[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            Mds[threadIdx.y][threadIdx.x] = 0.0f;

        if ((ph * TILE_WIDTH + threadIdx.y) < Width && Col < Width)
            Nds[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
        else
            Nds[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        C[Row * Width + Col] = Pvalue;
}

// Kernel using dynamic shared memory (runtime)
__global__ void matMulDynamic(float *A, float *B, float *C, int Width) {
    extern __shared__ float sharedMem[];
    
    float *Mds = sharedMem;
    float *Nds = &sharedMem[TILE_WIDTH * TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < Width && (ph * TILE_WIDTH + threadIdx.x) < Width)
            Mds[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            Mds[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        if ((ph * TILE_WIDTH + threadIdx.y) < Width && Col < Width)
            Nds[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
        else
            Nds[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[threadIdx.y * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        C[Row * Width + Col] = Pvalue;
}

// Function to initialize matrices
void initializeMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width * Width; i++) {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

// Function to measure kernel execution time
float measureKernelExecution(void (*kernel)(float *, float *, float *, int), float *A, float *B, float *C, float *d_A, float *d_B, float *d_C, int Width, size_t sharedMemSize = 0) {
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaMemcpy(d_A, A, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Width * Width * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (sharedMemSize > 0) {
        kernel<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, d_C, Width);
    } else {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, Width);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(C, d_C, Width * Width * sizeof(float), cudaMemcpyDeviceToHost);

    return elapsedTime;
}

int main() {
    int Width = 2048;  // Large matrix size
    size_t size = Width * Width * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_fixed = (float *)malloc(size);
    float *h_C_dynamic = (float *)malloc(size);

    // Initialize matrices
    initializeMatrix(h_A, Width);
    initializeMatrix(h_B, Width);

    // Allocate device memory
    float *d_A, *d_B, *d_C_fixed, *d_C_dynamic;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_fixed, size);
    cudaMalloc(&d_C_dynamic, size);

    // Query device properties
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    size_t maxSharedMem = devProp.sharedMemPerBlock; // Max shared memory per block

    // Measure execution times
    float fixedTime = measureKernelExecution(matMulFixed, h_A, h_B, h_C_fixed, d_A, d_B, d_C_fixed, Width);
    float dynamicTime = measureKernelExecution(matMulDynamic, h_A, h_B, h_C_dynamic, d_A, d_B, d_C_dynamic, Width, maxSharedMem);

    // Print performance comparison
    printf("\nMatrix Multiplication Performance Comparison\n");
    printf("------------------------------------------------\n");
    printf("| Method       | Execution Time (ms) | Shared Memory Usage |\n");
    printf("|-------------|--------------------|----------------------|\n");
    printf("| Fixed       | %10.4f ms        | %10lu B         |\n", fixedTime, TILE_WIDTH * TILE_WIDTH * 2 * sizeof(float));
    printf("| Dynamic     | %10.4f ms        | %10lu B         |\n", dynamicTime, maxSharedMem);
    printf("------------------------------------------------\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_fixed);
    free(h_C_dynamic);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_fixed);
    cudaFree(d_C_dynamic);

    return 0;
}
