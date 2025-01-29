#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 6
#define HEIGHT 6

// CUDA Kernel for a 3×3 image blur
__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height) {
    // Compute global row and column indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        // Iterate over 3×3 neighborhood
        for (int blurRow = -1; blurRow <= 1; ++blurRow) {
            for (int blurCol = -1; blurCol <= 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Ensure the thread is within bounds
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    int idx = curRow * width + curCol;
                    pixVal += input[idx]; // Sum up pixel values
                    pixels++; // Count valid pixels
                }
            }
        }

        // Compute the average and store the blurred pixel value
        int outIdx = row * width + col;
        output[outIdx] = pixVal / pixels;
    }
}

int main() {
    // Define and initialize a 6×6 grayscale image
    unsigned char hostInput[WIDTH * HEIGHT] = {
        10,  20,  30,  40,  50,  60,
        70,  80,  90, 100, 110, 120,
        130, 140, 150, 160, 170, 180,
        190, 200, 210, 220, 230, 240,
        250, 255, 245, 235, 225, 215,
        205, 195, 185, 175, 165, 155
    };

    unsigned char hostOutput[WIDTH * HEIGHT]; // Output image

    // Allocate device memory
    unsigned char *deviceInput, *deviceOutput;
    cudaMalloc((void **)&deviceInput, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc((void **)&deviceOutput, WIDTH * HEIGHT * sizeof(unsigned char));

    // Copy image data to GPU
    cudaMemcpy(deviceInput, hostInput, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define 2D grid and 2D block dimensions
    dim3 blockDim(2, 2); // Each block contains 2×2 threads
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch the blur kernel
    blurKernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, WIDTH, HEIGHT);

    // Copy blurred image back to host
    cudaMemcpy(hostOutput, deviceOutput, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print the blurred image
    printf("Blurred Image:\n");
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            printf("%3d ", hostOutput[row * WIDTH + col]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}
