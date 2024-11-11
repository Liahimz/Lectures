#ifndef CUSTOM_PROCESSOR
#define CUSTOM_PROCESSOR

#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


class CustomImageProcessor {
public:
    CustomImageProcessor():   ksize{1},
                              scale{1},
                              delta{0},
                              ddepth{3} {};

    cv::Mat process_image(cv::Mat& input_image);
private:
    cv::Mat src;
    cv::Mat src_gray;
    int ksize;
    int scale;
    int delta;
    int ddepth;
    cv::Mat grad;
};

#endif