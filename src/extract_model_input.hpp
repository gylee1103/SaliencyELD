#ifndef EXTRACT_MODEL_INPUT
#define EXTRACT_MODEL_INPUT
#include <vector>
#include <opencv2/core/core.hpp>
#include <caffe/proto/caffe.pb.h>
#include "common.hpp"
#include "region.hpp"

/* ---------------------------------------------------------------------
 * Generate Datum of Initial Feature Distance Maps of given query region.
 * ---------------------------------------------------------------------*/
void GenerateInitialFeatureDistanceMapDatum(RegionInfos& region_infos,
        GridToRegion& grid2region, const Region& target_region,
        caffe::Datum& output);

/* ---------------------------------------------------------
 * Generate SLIC region information using vl-feat library.
 * ---------------------------------------------------------*/
void GenerateSlicRegions(cv::Mat& image,
        RegionInfos& region_infos, GridToRegion& grid2region);

/* ---------------------------------------------------
 * Initialize low-level features of each SLIC region.
 * ---------------------------------------------------*/
void InitializeLowlevelFeatures(cv::Mat& image, RegionInfos& region_infos);

/* ----------------------------------------------
 * Convert Image to Datum for VGG16 input
 * ----------------------------------------------*/
void GenerateImageDatum(cv::Mat& _cv_img,
        std::vector<caffe::Datum>& img_vec, const int crop_size);

#endif
