#include <opencv2/opencv.hpp>
#include "image_processor.h"

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    cv::Mat grayscaleImage = ImageProcessor::convertToGrayscale(image);
    cv::imwrite("output.jpg", grayscaleImage);

    return 0;
}
