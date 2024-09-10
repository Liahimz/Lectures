#include <opencv2/opencv.hpp>  // OpenCV header
#include "EigenTest.hpp"         // Custom header for Eigen usage

int main() {
    // OpenCV example usage
    cv::Mat image = cv::imread("testdata/image.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }


    // Show the original image
    cv::imshow("Original Image", image);
    cv::waitKey(0);

    // Convert the image to CV_32F for floating point operations
    image.convertTo(image, CV_32F);

    // Create an instance of MyEigen and perform operations
    MyEigen example;
    example.performEigenOperations(image);

    // Convert the image back to CV_8UC3 for display
    image.convertTo(image, CV_8UC3);

    // Show the modified image
    cv::imshow("Modified Image", image);
    cv::waitKey(0);

    return 0;
}
