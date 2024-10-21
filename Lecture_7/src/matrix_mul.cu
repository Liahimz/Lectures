#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrixMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    size_t size = N * N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch the kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 16;  // Define matrix size NxN
    std::vector<float> A(N * N, 1.0f);  // Matrix A filled with 1s
    std::vector<float> B(N * N, 2.0f);  // Matrix B filled with 2s
    std::vector<float> C(N * N, 0.0f);  // Result matrix C filled with 0s

    matrixMul(A, B, C, N);

    // Display part of the result matrix
    std::cout << "Result matrix C (first row):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
