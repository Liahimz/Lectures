#include "image_processor.hpp"
#include <iostream>

cv::Mat CustomImageProcessor::process_image(cv::Mat& input_image) {
// Check if image is loaded fine
  if( input_image.empty() )
  {
    std::cout << "Error opening image: " << std::endl;
  }
  GaussianBlur(input_image, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

  cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;
  Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
  Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);

  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

  return grad;
}


