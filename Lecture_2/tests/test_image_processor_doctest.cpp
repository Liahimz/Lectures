#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "image_processor.h"

TEST_CASE("Testing convertToGrayscale") {
    cv::Mat colorImage = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat grayscaleImage = ImageProcessor::convertToGrayscale(colorImage);

    CHECK(grayscaleImage.type() == CV_8UC1);
    CHECK(grayscaleImage.rows == colorImage.rows);
    CHECK(grayscaleImage.cols == colorImage.cols);
}
