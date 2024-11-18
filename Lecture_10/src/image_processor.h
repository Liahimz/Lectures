#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>

class ImageProcessor {
public:
    static cv::Mat convertToGrayscale(const cv::Mat& inputImage);
};

#endif // IMAGE_PROCESSOR_H
