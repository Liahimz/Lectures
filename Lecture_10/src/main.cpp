#include <opencv2/opencv.hpp>
#include "image_processor.h"

int main(int argc, char** argv) {
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::imshow("Input Image", image);
    cv::waitKey(0);

    cv::Mat grayscaleImage = ImageProcessor::convertToGrayscale(image);
    cv::imshow("Grayscale Image", grayscaleImage);
    cv::waitKey(0);

    return 0;
}
