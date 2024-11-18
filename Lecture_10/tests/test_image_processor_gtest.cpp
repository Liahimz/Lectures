#include <string>
#include <gtest/gtest.h>
#include "image_processor.h"

TEST(ImageProcessorTest, ConvertToGrayscale) {
    cv::Mat colorImage = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat grayscaleImage = ImageProcessor::convertToGrayscale(colorImage);

    ASSERT_EQ(grayscaleImage.type(), CV_8UC1);
    ASSERT_EQ(grayscaleImage.rows, colorImage.rows);
    ASSERT_EQ(grayscaleImage.cols, colorImage.cols);
    ASSERT_NE(grayscaleImage.type(), CV_8UC3);
    ASSERT_TRUE(grayscaleImage.rows == 100);
    std::string str("xyz");
    str += "ABC";
    ASSERT_STRCASEEQ(str.c_str(), "XYZabc");
}
