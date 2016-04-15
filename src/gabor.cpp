#include <opencv2/imgproc/imgproc.hpp>
#include "gabor.hpp"
using std::vector;

void GenerateGaborKernels(vector<cv::Mat> &gb_kernels) {
    const int kernel_size = 11;
    enum {
        SIGMA = 0,
        LAMBDA = 1,
        PSI = 2,
    };
    static const double gabor_parameters[3][3] =
    {
        {5, 0.7, 90}, {5, 1, 90}, {5, 1.3, 90}
    };
    static const double theta_parameters[8] =
        {0, 24, 48, 72, 96, 120, 144, 168};

    int hks = (kernel_size - 1) / 2;
    double del = 2.0 / (kernel_size - 1);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 8; j++) {
            double theta = theta_parameters[j] * CV_PI/180;
            double psi = gabor_parameters[i][PSI] * CV_PI/180;
            double lambda = gabor_parameters[i][LAMBDA];
            double sigma = gabor_parameters[i][SIGMA] / kernel_size;
            double x_theta;
            double y_theta;
            cv::Mat kernel(kernel_size, kernel_size, CV_32F);
            for (int y = -hks; y <= hks; y++) {
                for (int x = -hks; x <= hks; x++) {
                    double x_theta = x * del * cos(theta) + y * del * sin(theta);
                    double y_theta = -x * del * sin(theta) + y * del * cos(theta);
                    kernel.at<float>(hks + y, hks + x) =
                        static_cast<float>(
                            exp(-0.5 * (pow(x_theta, 2) +
                                        pow(y_theta, 2)) / pow(sigma, 2))
                            * cos(2 * CV_PI * x_theta / lambda + psi));
                }
            }
            gb_kernels.push_back(kernel);
        }
    }
}

void CalculateGaborFilterdImages(const cv::Mat &image, 
        vector<cv::Mat> &gb_filtered_images) {
    cv::Mat gray_image;
    cv::Mat gimage_f;
    cv::cvtColor(image, gray_image, CV_BGR2GRAY);
    gray_image.convertTo(gimage_f, CV_32F, 1.0/255, 0);

    vector<cv::Mat> gb_kernels;
    GenerateGaborKernels(gb_kernels);
    for(vector<cv::Mat>::iterator it = gb_kernels.begin(); it != gb_kernels.end(); it++) {
        cv::Mat dest;
        cv::filter2D(gimage_f, dest, CV_32F, *it);
        dest = cv::abs(dest);
        cv::Mat tmp;
        dest.convertTo(tmp, CV_8U, 255, 0);
        cv::medianBlur(tmp, tmp, 15);
        tmp.convertTo(dest, CV_32FC1, 1/255.0);
        gb_filtered_images.push_back(dest);
    }
}
