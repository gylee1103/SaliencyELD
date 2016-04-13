#include <algorithm>
#include <boost/chrono.hpp>
#include "region.hpp"

Region::Region() {
	_is_initialized = false;
    _InitializeMemvs();
}

void Region::_InitializeMemvs() {
	_mean_label = 0;
	_center_col = 0;
	_center_row = 0;
	for (int i = 0; i < DIM_COLOR; i++) {
		_mean_color_bgr[i] = 0;
		_mean_color_lab[i] = 0;
		_mean_color_hsv[i] = 0;
		for (int j = 0; j < DIM_HIST_BIN; j++) {
			_histo_bgr[i][j] = 0;
			_histo_lab[i][j] = 0;
			_histo_hsv[i][j] = 0;
		}
	}
	for (int i = 0; i < DIM_GABOR_BIN; i++) {
		_histo_gabor[i] = 0;
	}
}

void Region::Initialize(const Mat& bgr_image,
        const Mat& lab_image, const Mat& hsv_image) {

	_InitializeMemvs();
	int bin;
	int bin_size = (CVIMAGE_MAX + 1) / DIM_HIST_BIN;

	for (auto pit = _pixels.begin(); pit != _pixels.end(); ++pit) {
		_mean_label += static_cast<float>(pit->label);
		_center_row += static_cast<float>(pit->row);
		_center_col += static_cast<float>(pit->col);
		for (int i = 0; i < DIM_COLOR; i++) {
			float value;
			// BGR space
			value = static_cast<float>(bgr_image.at<cv::Vec3b>(
                        pit->row, pit->col)[i]);
			_mean_color_bgr[i] += value / CVIMAGE_MAX; // Mean color
			bin = value / bin_size;
			_histo_bgr[i][bin] += 1; // Histogram

			// LAB space
			value = static_cast<float>(lab_image.at<cv::Vec3b>(
                        pit->row, pit->col)[i]);
			_mean_color_lab[i] += value / CVIMAGE_MAX;
			bin = value / bin_size;
			_histo_lab[i][bin] += 1;

			// HSV space
			value = static_cast<float>(hsv_image.at<cv::Vec3b>(
                        pit->row, pit->col)[i]);
			_mean_color_hsv[i] += value / CVIMAGE_MAX;
			bin = value / bin_size;
			_histo_hsv[i][bin] += 1;
		}
	}
	_region_size = _pixels.size(); // the number of pixels
	
	_mean_label /= _region_size; // Mean of Label(salient or not)
	_center_row /= _region_size; // Location of center pixel
	_center_col /= _region_size;
	_normalized_row = _center_row / bgr_image.rows; // Normalized location
	_normalized_col = _center_col / bgr_image.cols;

	for (int i = 0; i < DIM_COLOR; i++) {
		_mean_color_bgr[i] /= _region_size;
		_mean_color_lab[i] /= _region_size;
		_mean_color_hsv[i] /= _region_size;

		for (int j = 0; j < DIM_HIST_BIN; j++) {
			_histo_bgr[i][j] /= _region_size;
			_histo_lab[i][j] /= _region_size;
			_histo_hsv[i][j] /= _region_size;
		}
	}
	
	_is_initialized = true;
}

void Region::put_gabor_values(const vector<Mat> &gb_filtered_images) {
	for (auto pit = _pixels.begin(); pit != _pixels.end(); ++pit) {
		for (int gbindex = 0; gbindex < DIM_GABOR_BIN; gbindex++) {
			_histo_gabor[gbindex] += std::min((float)1.0, 
                    gb_filtered_images[gbindex].at<float>(pit->row, pit->col));
		}
	}
	_max_gabor = 0;
	for (int gbindex = 0; gbindex < DIM_GABOR_BIN; gbindex++) {
		_histo_gabor[gbindex] /= _region_size;
		if (_max_gabor < _histo_gabor[gbindex])
			_max_gabor = _histo_gabor[gbindex];
	}
}
