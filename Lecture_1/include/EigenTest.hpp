#ifndef MYEIGEN_HPP
#define MYEIGEN_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp> 

class MyEigen {
public:
    void convertCV2MAT(const cv::Mat& src, std::vector<Eigen::MatrixXf>& dst);

    void convertMAT2CV(const std::vector<Eigen::MatrixXf>& src, cv::Mat& dst);
    
    void performEigenOperations(cv::Mat& src);
private:
    std::vector<Eigen::MatrixXf> inner_storage;
};

#endif