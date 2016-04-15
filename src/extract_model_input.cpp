#include <cassert>
#include <cmath>
#include <string>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
extern "C" {
#include <vl/generic.h>
#include <vl/slic.h>
}
#include "extract_model_input.hpp"
#include "gabor.hpp"

float GetHistogramDiff(const Hist& other_hist_color,
        const Hist& target_hist_color, int index) {
    float sum = 0;
    float eps = 0.00000001; // prevent divide zero
    for (int b = 0; b < DIM_HIST_BIN; b++) {
        sum += 2 * pow(other_hist_color[index][b] - 
                target_hist_color[index][b], 2) / (other_hist_color[index][b] + 
                target_hist_color[index][b] + eps);
    }
    return sum/4.0;
}

/* --------------------------------------------------------------
 * Extract the initial feature distance map of the query region.
 * -------------------------------------------------------------*/
void FillInitialFeatureDistance(const std::pair<const int, Region>& creg_pair,
        const Region &query_region, GridToRegion& grid2region,
        float input_map[DIM_MODEL_FEED][GRID_SIZE][GRID_SIZE]) {
    vector<std::pair<int, int> > dominate_gcells;
    for (int gr = 0; gr < GRID_SIZE; gr++) {
        for (int gc = 0; gc < GRID_SIZE; gc++) {
            if (grid2region[gr][gc] == creg_pair.first) {
                dominate_gcells.push_back(std::make_pair(gr, gc));
            }
        }
    }

    if (dominate_gcells.size() == 0) return;
    auto& current_region = creg_pair.second;

    float init_features[DIM_MODEL_FEED];
    /* -----------------------------------------
     * Calculate Low-level properties(including difference)
     * -----------------------------------------*/
    int index = 0;
    for (int cs = 0; cs < 3; cs++) { // see common.hpp enum
        // cs(color space) => 0: RGB 1: LAB 2: HSV
        const Hist& it_hist = current_region.get_color_histogram(cs);
        const Hist& target_hist = query_region.get_color_histogram(cs);
        for (int i = 0; i < DIM_COLOR; i++) {
            // Color distance
            init_features[index] =
                current_region.get_mean_color(cs)[i] -
                query_region.get_mean_color(cs)[i];
            index++;
            // Color 
            init_features[index] =
                current_region.get_mean_color(cs)[i] - 0.5;
            index++;
        }
        for (int i = 0; i < DIM_COLOR; i++) {
            // Color Histogram distance (CHI-distance)
            init_features[index] =
                GetHistogramDiff(it_hist, target_hist, i);
            index++;
        }
    }
    // Gabor Histogram and its raw distance
    for (int gb = 0; gb < DIM_GABOR_BIN; gb++) {
        init_features[index] = 
            current_region.get_gabor_values()[gb] -
            query_region.get_gabor_values()[gb];
        index++;
    }
    init_features[index] = //(1dims)
        current_region.get_max_gabor() - query_region.get_max_gabor();
    index++;
    // location difference (2dims)
    init_features[index] =
        current_region.get_normalized_row() - query_region.get_normalized_row();
    index++;
    init_features[index] =
        current_region.get_normalized_col() - query_region.get_normalized_col();
    index++;
    assert(index == DIM_MODEL_FEED);

    /*--------------------------------------------
     * now put features to dominated grid-cell
     *-------------------------------------------*/
    for (const auto& pit: dominate_gcells) {
        int row = pit.first;
        int col = pit.second;
        for (int index = 0; index < DIM_MODEL_FEED; index++) {
                input_map[index][row][col] = init_features[index];
        }
    }
}

/* ---------------------------------------------------------------------
 * Generate Datum of Initial Feature Distance Maps of given query region.
 * ---------------------------------------------------------------------*/
void GenerateInitialFeatureDistanceMapDatum(RegionInfos& region_infos,
        GridToRegion& grid2region, const Region& query_region, caffe::Datum& output) {
    float tmp_float_map[DIM_MODEL_FEED][GRID_SIZE][GRID_SIZE] = {0};
    for (const auto& creg_pair: region_infos) {
        FillInitialFeatureDistance(creg_pair, query_region,
                grid2region, tmp_float_map);
    }
    // Copy tmp_map to output Datum
    output.set_channels(DIM_MODEL_FEED);
    output.set_height(GRID_SIZE);
    output.set_width(GRID_SIZE);
    output.set_label(0);
    output.clear_data();
    output.clear_float_data();
    for (int c = 0; c < DIM_MODEL_FEED; c++) {
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int w = 0; w < GRID_SIZE; w++) {
                output.add_float_data(tmp_float_map[c][r][w]);
            }
        }
    }
}

/* ---------------------------------------------------------
 * Generate SLIC region information using vl-feat library.
 * ---------------------------------------------------------*/
void GenerateSlicRegions(Mat& image, RegionInfos& region_infos,
                         GridToRegion& grid2region) {
    const int width = image.cols;
    const int height = image.rows;
    const int MAX_REGION_INDEX = 1000; // MAX number of slic superpixels
    const int SLIC_REGULARIZER = 1000;
    const int MIN_SLIC_SIZE = 30;
    const int INIT_SLIC_SIZE = sqrt(width*height/400);

    // Prepare data and call vl-feat SLIC
    float* input = new float[height*width*DIM_COLOR]();
    Mat labels(image.size(), CV_32SC1);
    
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);
            *(input + row*width + col) = intensity[0];
            *(input + width*height + row*width + col) = intensity[1];
            *(input + width*height*2 + row*width + col) = intensity[2];
        }
    }
    vl_slic_segment(labels.ptr<vl_uint32>(), input,
            width, height, image.channels(), INIT_SLIC_SIZE, SLIC_REGULARIZER,
            MIN_SLIC_SIZE);

    // Grouping pixels by their region label.
    float gc_width = width / (float) GRID_SIZE;
    float gc_height = height / (float) GRID_SIZE;
    for (int i = 0; i < width*height; i++) {
        int w = i % width;
        int h = i / width;
        int slic_label = static_cast<int>(labels.at<unsigned int>(h, w));
        auto it = region_infos.find(slic_label);
        // Generate new region index
        if(it == region_infos.end()) { 
            it = region_infos.insert(
                    std::pair<int, Region>(slic_label, Region())).first;
        }
        it->second.put_pixel(Pixel(h, w, 1));
    }

  #pragma omp parallel for
    for (int hidx = 0; hidx < GRID_SIZE; hidx++) {
        for (int widx = 0; widx < GRID_SIZE; widx++) {
            int cnt[MAX_REGION_INDEX] = {0};
            for (int h = hidx*gc_height; h < (int)((hidx+1)*gc_height); h++) {
                for (int w = widx*gc_width; w < (int)((widx+1)*gc_width); w++) {
                    int label = static_cast<int>(labels.at<unsigned int>(h, w));
                    cnt[label]++;
                }
            }
            int dom_sindx = -1;
            int dom_cnt = -1;
            for (int j = 0; j < MAX_REGION_INDEX; j++) {
                if (dom_cnt < cnt[j]) {
                    dom_sindx = j;
                    dom_cnt = cnt[j];
                }
            }
            grid2region[hidx][widx] = dom_sindx;
        }
    }
    delete (input);
}

/* ---------------------------------------------------
 * Initialize low-level features of each SLIC region.
 * ---------------------------------------------------*/
void InitializeLowlevelFeatures(Mat& image, RegionInfos& region_infos) {
    Mat lab_image;
    Mat hsv_image;
    cv::cvtColor(image, lab_image, CV_BGR2Lab);
    cv::cvtColor(image, hsv_image, CV_BGR2HSV);
    vector<Mat> gb_filtered_images;
    CalculateGaborFilterdImages(image, gb_filtered_images);
  #pragma omp parallel
  #pragma omp single
  {
    for (auto it = std::begin(region_infos);
            it != std::end(region_infos); it++) {
  #pragma omp task firstprivate(it)
      {
        it->second.Initialize(image, lab_image, hsv_image);
        it->second.put_gabor_values(gb_filtered_images);
      }
    }
  #pragma omp taskwait
  }
}

/* ----------------------------------------------
 * Convert Image to Datum for VGG16 input
 * ----------------------------------------------*/
void GenerateImageDatum(Mat& _cv_img, vector<caffe::Datum>& img_vec, 
                        const int crop_size) {
    caffe::Datum datum;
    Mat resized_img;
    resize(_cv_img, resized_img, cv::Size(crop_size, crop_size));
    img_vec.push_back(caffe::Datum());
    img_vec[0].set_height(crop_size);
    img_vec[0].set_channels(3);
    img_vec[0].set_width(crop_size);
    img_vec[0].clear_data();
    img_vec[0].clear_float_data();
    std::string buffer(3*crop_size*crop_size, 0);
    img_vec[0].set_data(buffer);
    std::string* data_ptr = img_vec[0].mutable_data();
    for (int h = 0; h < resized_img.rows; h++) {
        const uchar* ptr = resized_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < resized_img.cols; w++) {
            for (int c = 0; c < resized_img.channels(); c++) {
                int datum_index = (c * crop_size + h) * crop_size + w;
                (*data_ptr)[datum_index] = static_cast<char>(ptr[img_index++]);
            }
        }
    }
}
