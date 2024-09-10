#include "EigenTest.hpp"
#include <iostream>


// Convert cv::Mat to a vector of Eigen::MatrixXf (for each channel)
void MyEigen::convertCV2MAT(const cv::Mat& src, std::vector<Eigen::MatrixXf>& dst) {
    // Split the multi-channel cv::Mat into separate channels
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);

    // Resize the destination vector to hold 3 Eigen matrices
    dst.resize(3);

    // Convert each channel to Eigen::MatrixXf
    for (int c = 0; c < 3; ++c) {
        dst[c].resize(channels[c].rows, channels[c].cols);
        for (int i = 0; i < channels[c].rows; ++i) {
            for (int j = 0; j < channels[c].cols; ++j) {
                // Assuming src is of type CV_32F, adjust if necessary
                dst[c](i, j) = channels[c].at<float>(i, j);
            }
        }
    }
}

// Convert a vector of Eigen::MatrixXf back to cv::Mat
void MyEigen::convertMAT2CV(const std::vector<Eigen::MatrixXf>& src, cv::Mat& dst) {
    // Create a vector of cv::Mat to hold each channel
    std::vector<cv::Mat> channels(3);

    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(src[c].rows(), src[c].cols(), CV_32F);
        for (int i = 0; i < src[c].rows(); ++i) {
            for (int j = 0; j < src[c].cols(); ++j) {
                channels[c].at<float>(i, j) = src[c](i, j);
            }
        }
    }

    // Merge the channels back into a single 3-channel cv::Mat
    cv::merge(channels, dst);
}


// Perform Eigen operations on the image (multiply by a scalar)
void MyEigen::performEigenOperations(cv::Mat& src) {
    // Create a copy of the source image
    cv::Mat src_copy(src);

    // Convert cv::Mat to Eigen matrices (for each channel)
    convertCV2MAT(src_copy, inner_storage);

    // Multiply each Eigen matrix (channel) by a scalar
    double scalar = 0.5;  // Multiply by 2
    for (auto& matrix : inner_storage) {
        matrix *= scalar;  // Scalar multiplication
    }

    // Convert Eigen matrices back to cv::Mat
    convertMAT2CV(inner_storage, src);
}
