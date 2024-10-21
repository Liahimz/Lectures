#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA kernel for convolution operation
__global__ void conv2d(float* input, float* kernel, float* output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputHeight && col < outputWidth) {
        float sum = 0.0f;

        // Apply convolution (sliding window)
        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row + i;
                int inputCol = col + j;
                sum += input[inputRow * inputWidth + inputCol] * kernel[i * kernelWidth + j];
            }
        }
        output[row * outputWidth + col] = sum;
    }
}

// CUDA kernel for fully connected layer (matrix-vector multiplication)
__global__ void linearLayer(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = bias[idx];  // Start with bias
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = sum;
    }
}

// Function to perform convolution followed by a fully connected layer
void convNet(float* input, float* kernel, float* convOutput, float* fcWeights, float* fcBias, float* finalOutput,
             int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight,
             int fcInputSize, int fcOutputSize) {

    // Convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2d<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, convOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, outputWidth, outputHeight);

    // Fully connected layer kernel
    int threadsPerBlockFC = 256;
    int blocksPerGridFC = (fcOutputSize + threadsPerBlockFC - 1) / threadsPerBlockFC;
    linearLayer<<<blocksPerGridFC, threadsPerBlockFC>>>(convOutput, fcWeights, fcBias, finalOutput, fcInputSize, fcOutputSize);
}

int main() {
    // Load grayscale image using OpenCV (e.g., MNIST image)
    cv::Mat img = cv::imread("/home/mzingerenko/Desktop/Lectures/Lecture_7/data/mnist_img.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // Resize image if necessary (should be 28x28)
    int inputWidth = 28, inputHeight = 28;
    if (img.size().width != 28 || img.size().height != 28) {
        cv::resize(img, img, cv::Size(28, 28));
    }

    // Normalize image (0 to 1 range)
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Convert OpenCV Mat to a flat vector for CUDA
    std::vector<float> input(inputWidth * inputHeight);
    std::memcpy(input.data(), img.data, input.size() * sizeof(float));

    // Define kernel and layer sizes
    int kernelWidth = 5, kernelHeight = 5;  // 5x5 convolution kernel
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    // Fully connected layer dimensions
    int fcInputSize = outputWidth * outputHeight;
    int fcOutputSize = 10;  // 10 output classes (for digit classification)

    // Allocate memory for kernel, convolution output, weights, bias, and final output
    std::vector<float> kernel(kernelWidth * kernelHeight, 0.1f);  // Simple 5x5 kernel with 0.1 values
    std::vector<float> convOutput(outputWidth * outputHeight, 0.0f);  // Convolution output
    std::vector<float> fcWeights(fcOutputSize * fcInputSize, 0.1f);  // Fully connected weights
    std::vector<float> fcBias(fcOutputSize, 0.0f);  // Fully connected biases
    std::vector<float> finalOutput(fcOutputSize, 0.0f);  // Final output

    // Device memory pointers
    float *d_input, *d_kernel, *d_convOutput, *d_fcWeights, *d_fcBias, *d_finalOutput;

    // Allocate device memory
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
    cudaMalloc(&d_convOutput, convOutput.size() * sizeof(float));
    cudaMalloc(&d_fcWeights, fcWeights.size() * sizeof(float));
    cudaMalloc(&d_fcBias, fcBias.size() * sizeof(float));
    cudaMalloc(&d_finalOutput, finalOutput.size() * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcWeights, fcWeights.data(), fcWeights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcBias, fcBias.data(), fcBias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution and fully connected layer
    convNet(d_input, d_kernel, d_convOutput, d_fcWeights, d_fcBias, d_finalOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, outputWidth, outputHeight, fcInputSize, fcOutputSize);

    // Copy final output back to host
    cudaMemcpy(finalOutput.data(), d_finalOutput, finalOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Display final output (predictions)
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < fcOutputSize; ++i) {
        std::cout << "Class " << i << ": " << finalOutput[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_convOutput);
    cudaFree(d_fcWeights);
    cudaFree(d_fcBias);
    cudaFree(d_finalOutput);

    return 0;
}

