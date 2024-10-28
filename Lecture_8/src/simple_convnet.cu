#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <fstream>

// Function to read kernel weights from file
void readKernelWeights(const std::string& filename, std::vector<float>& kernels, int kernelSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filename << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (file >> value && count < kernelSize) {
        kernels.push_back(value);
        count++;
    }

    if (count != kernelSize) {
        std::cerr << "Warning: Expected in Kernel " << kernelSize << " values, but found " << count << std::endl;
    }
    file.close();
}

// Function to read fully connected layer weights and biases from file
void readFullyConnectedWeights(const std::string& weightFile, const std::string& biasFile, 
                               std::vector<float>& fcWeights, std::vector<float>& fcBias, 
                               int fcWeightSize, int fcBiasSize) {
    // Read fully connected layer weights
    std::ifstream weightStream(weightFile);
    if (!weightStream.is_open()) {
        std::cerr << "Error: Could not open the weight file: " << weightFile << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (weightStream >> value && count < fcWeightSize) {
        fcWeights.push_back(value);
        count++;
    }

    if (count != fcWeightSize) {
        std::cerr << "Warning: Expected in FC Weights " << fcWeightSize << " weights, but found " << count << std::endl;
    }
    weightStream.close();

    // Read fully connected layer biases
    std::ifstream biasStream(biasFile);
    if (!biasStream.is_open()) {
        std::cerr << "Error: Could not open the bias file: " << biasFile << std::endl;
        return;
    }

    count = 0;
    while (biasStream >> value && count < fcBiasSize) {
        fcBias.push_back(value);
        count++;
    }

    if (count != fcBiasSize) {
        std::cerr << "Warning: Expected in FC Bias" << fcBiasSize << " biases, but found " << count << std::endl;
    }
    biasStream.close();
}

// CUDA kernel for convolution operation with multiple output channels
__global__ void conv2d(float* input, float* kernels, float* output, int inputWidth, int inputHeight, 
                       int kernelWidth, int kernelHeight, int outputWidth, int outputHeight, int numKernels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    if (row < outputHeight && col < outputWidth && channel < numKernels) {
        float sum = 0.0f;

        // Apply convolution for the specific kernel (output channel)
        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row + i;
                int inputCol = col + j;
                sum += input[inputRow * inputWidth + inputCol] * kernels[channel * kernelWidth * kernelHeight + i * kernelWidth + j];
            }
        }
        // Store the result in the output tensor (each channel has its own slice in output)
        output[channel * outputHeight * outputWidth + row * outputWidth + col] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluActivation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0) data[idx] = 0;
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
void convNet(float* input, float* kernels, float* convOutput, float* fcWeights, float* fcBias, float* finalOutput,
             int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight,
             int fcInputSize, int fcOutputSize, int numKernels) {

    // Convolution kernel with multiple output channels
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, numKernels);
    conv2d<<<blocksPerGrid, threadsPerBlock>>>(input, kernels, convOutput, inputWidth, inputHeight, 
                                               kernelWidth, kernelHeight, outputWidth, outputHeight, numKernels);

    // ReLU activation
    int convOutputSize = outputWidth * outputHeight * numKernels;
    int threadsPerBlockReLU = 256;
    int blocksPerGridReLU = (convOutputSize + threadsPerBlockReLU - 1) / threadsPerBlockReLU;
    reluActivation<<<blocksPerGridReLU, threadsPerBlockReLU>>>(convOutput, convOutputSize);

    // Fully connected layer kernel
    int threadsPerBlockFC = 256;
    int blocksPerGridFC = (fcOutputSize + threadsPerBlockFC - 1) / threadsPerBlockFC;
    linearLayer<<<blocksPerGridFC, threadsPerBlockFC>>>(convOutput, fcWeights, fcBias, finalOutput, fcInputSize, fcOutputSize);
}

int main() {
    // Load grayscale image using OpenCV (e.g., MNIST image)
    cv::Mat img = cv::imread("/workspace/data/mnist_img3.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // Resize image if necessary (should be 28x28)
    int inputWidth = 28, inputHeight = 28;
    if (img.cols != 28 || img.rows != 28) {
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
    int numKernels = 6;  // Number of output channels

    // Fully connected layer dimensions
    int fcInputSize = outputWidth * outputHeight * numKernels;
    int fcOutputSize = 10;  // 10 output classes (for digits 0-9)

    // // Random number generation
    // std::default_random_engine generator;
    // std::normal_distribution<float> distribution(0.0f, 0.1f);  // Mean 0, stddev 0.1

    // // Initialize kernels for 6 output channels with random values
    // std::vector<float> kernels(kernelWidth * kernelHeight * numKernels);
    // for (auto& weight : kernels) {
    //     weight = distribution(generator);
    // }
    // // Initialize fully connected layer weights and biases with random values
    // std::vector<float> fcWeights(fcOutputSize * fcInputSize);
    // for (auto& weight : fcWeights) {
    //     weight = distribution(generator);
    // }

    // std::vector<float> fcBias(fcOutputSize);
    // for (auto& bias : fcBias) {
    //     bias = distribution(generator);
    // }


    std::string weightsPath = "/workspace/data/";

    int32_t kernelSize = kernelWidth * kernelHeight * numKernels;
    // Load kernel weights
    std::vector<float> kernels;
    readKernelWeights(weightsPath + "conv_kernels.txt", kernels, kernelSize);

    int32_t fcWeightSize = fcOutputSize * fcInputSize;
    int32_t fcBiasSize = fcOutputSize;
    // Load fully connected layer weights and biases
    std::vector<float> fcBias;
    std::vector<float> fcWeights;
    readFullyConnectedWeights(weightsPath + "fc_weights.txt", weightsPath + "fc_biases.txt", fcWeights, fcBias, fcWeightSize, fcBiasSize);
    
    std::vector<float> convOutput(outputWidth * outputHeight * numKernels, 0.0f);  // Convolution output

    std::vector<float> finalOutput(fcOutputSize, 0.0f);  // Final output

    // Device memory pointers
    float *d_input, *d_kernels, *d_convOutput, *d_fcWeights, *d_fcBias, *d_finalOutput;

    // Allocate device memory
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_kernels, kernels.size() * sizeof(float));
    cudaMalloc(&d_convOutput, convOutput.size() * sizeof(float));
    cudaMalloc(&d_fcWeights, fcWeights.size() * sizeof(float));
    cudaMalloc(&d_fcBias, fcBias.size() * sizeof(float));
    cudaMalloc(&d_finalOutput, finalOutput.size() * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels.data(), kernels.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcWeights, fcWeights.data(), fcWeights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcBias, fcBias.data(), fcBias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution and fully connected layer
    convNet(d_input, d_kernels, d_convOutput, d_fcWeights, d_fcBias, d_finalOutput, 
            inputWidth, inputHeight, kernelWidth, kernelHeight, outputWidth, outputHeight, 
            fcInputSize, fcOutputSize, numKernels);

    // Copy final output back to host
    cudaMemcpy(finalOutput.data(), d_finalOutput, finalOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Apply softmax to interpret outputs as probabilities
    float maxVal = *std::max_element(finalOutput.begin(), finalOutput.end());
    float sum = 0.0f;
    for (auto& val : finalOutput) {
        val = std::exp(val - maxVal);  // For numerical stability
        sum += val;
    }
    for (auto& val : finalOutput) {
        val /= sum;
    }

    // Display final output (predictions)
    std::cout << "Predictions (probabilities):" << std::endl;
    for (int i = 0; i < fcOutputSize; ++i) {
        std::cout << "Class " << i << ": " << finalOutput[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_convOutput);
    cudaFree(d_fcWeights);
    cudaFree(d_fcBias);
    cudaFree(d_finalOutput);

    return 0;
}
