#include "image_processor.h"

cv::Mat ImageProcessor::convertToGrayscale(const cv::Mat& inputImage) {
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    return grayscaleImage;
}
