#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Описание входного тензора
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 28, 28);

    // Описание выходного тензора после свёртки
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 26, 26); // после свёртки

    // Описание фильтра (ядра свёртки)
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 3); // Фильтр 3x3

    // Описание свёрточной операции
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Устанавливаем размер выходного тензора
    int batch_size, channels, height, width;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &batch_size, &channels, &height, &width);

    // Выделение памяти для тензоров
    float* d_input;
    float* d_output;
    float* d_filter;
    cudaMalloc(&d_input, 28 * 28 * sizeof(float));  // Входное изображение
    cudaMalloc(&d_output, 26 * 26 * sizeof(float)); // Выход после свёртки
    cudaMalloc(&d_filter, 3 * 3 * sizeof(float));   // Фильтр

    // Запуск свёрточной операции
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, d_output);

    // Применение функции активации (ReLU)
    cudnnActivationDescriptor_t activation_desc;
    cudnnCreateActivationDescriptor(&activation_desc);
    cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    cudnnActivationForward(cudnn, activation_desc, &alpha, output_desc, d_output, &beta, output_desc, d_output);

    // Линейный слой (fully connected)
    float* d_fc_input;
    float* d_fc_output;
    float* d_weights;
    cudaMalloc(&d_fc_input, 26 * 26 * sizeof(float));
    cudaMalloc(&d_fc_output, sizeof(float));   // один выход
    cudaMalloc(&d_weights, 26 * 26 * sizeof(float));  // вес для каждого выхода

    // Вычисление fully connected слоя (умножение)
    cudnnOpTensorDescriptor_t fc_desc;
    cudnnCreateOpTensorDescriptor(&fc_desc);
    cudnnSetOpTensorDescriptor(fc_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

    cudnnOpTensor(cudnn, fc_desc, &alpha, output_desc, d_output, &alpha, output_desc, d_weights, &beta, output_desc, d_fc_output);

    // Освобождение ресурсов
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_fc_input);
    cudaFree(d_fc_output);
    cudaFree(d_weights);
    cudnnDestroy(cudnn);

    return 0;
}
