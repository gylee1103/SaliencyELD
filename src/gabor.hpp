#ifndef GABOR_HPP
#define GABOR_HPP
#include <vector>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;
void CalculateGaborFilterdImages(const Mat &image, 
		vector<Mat> &gb_filtered_images);

#endif
