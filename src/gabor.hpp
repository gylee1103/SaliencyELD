#ifndef GABOR_HPP
#define GABOR_HPP
#include <vector>
#include <opencv2/core/core.hpp>

void CalculateGaborFilterdImages(const cv::Mat& image, 
        std::vector<cv::Mat>& gb_filtered_images);

#endif
