#include <opencv2/opencv.hpp>
#include "image_processor.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: process path-to-image" << std::endl;
        return -1;
    }
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    cv::Mat grayscaleImage = ImageProcessor::convertToGrayscale(image);
    cv::imwrite("grayscale_input.png", grayscaleImage);

    return 0;
}
