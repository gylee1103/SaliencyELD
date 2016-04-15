#ifndef REGION_HPP
#define REGION_HPP
#include <cassert>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include "common.hpp"
using std::vector;
using cv::Mat;
using std::map;

class Region {
private:
    bool _is_initialized;

    vector<Pixel> _pixels;
    float _center_row;
    float _center_col;
    float _mean_label;
    Color _mean_color_bgr;
    Color _mean_color_lab;
    Color _mean_color_hsv;
    Hist _histo_bgr;
    Hist _histo_lab;
    Hist _histo_hsv;
    float _histo_gabor[DIM_GABOR_BIN];
    float _max_gabor;
    float _normalized_row;
    float _normalized_col;
    float _region_size;

    void _InitializeMemvs();

public:
    Region();
    void Initialize(const Mat& bgr_image,
            const Mat& lab_image, const Mat& hsv_image);

    inline bool is_empty() const {
        return _pixels.empty();
    }

// Putter functions.

    inline void put_pixel(const Pixel& new_pixel) {
        _pixels.push_back(new_pixel);
    }

    void put_gabor_values(const vector<Mat> &gb_filtered_images);

// Getter functions.

    inline float get_label() const {
        return _mean_label;
    }

    inline const vector<Pixel>& get_pixels() const {
        return _pixels;
    }
    
    inline const Color& get_mean_color(int color_space) const {
        switch(color_space) {
            case BGR_SPACE:
                return _mean_color_bgr;
            case LAB_SPACE:
                return _mean_color_lab;
            case HSV_SPACE:
                return _mean_color_hsv;
            default:
                break;
        }
        assert(false);
        return _mean_color_bgr;
    }
    
    inline const Hist& get_color_histogram(int color_space) const {
        switch(color_space) {
            case BGR_SPACE:
                return _histo_bgr;
            case LAB_SPACE:
                return _histo_lab;
            case HSV_SPACE:
                return _histo_hsv;
            default:
                break;
        }
        assert(false);
        return _histo_bgr;
    }

    inline float get_normalized_row() const {
        return _normalized_row;
    }

    inline float get_normalized_col() const {
        return _normalized_col;
    }

    inline const float* get_gabor_values() const {
        return _histo_gabor;
    }
    inline float get_max_gabor() const {
        return _max_gabor;
    }
};

typedef map<int, Region> RegionInfos;
#endif
