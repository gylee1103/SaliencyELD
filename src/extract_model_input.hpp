#ifndef EXTRACT_MODEL_INPUT
#define EXTRACT_MODEL_INPUT
#include <vector>
#include <opencv2/core/core.hpp>
#include "common.hpp"
#include "region.hpp"

using std::vector;
using cv::Mat;


/* ---------------------------------------------------------------------
 * Generate Datum of Initial Feature Distance Maps of given query region.
 * ---------------------------------------------------------------------*/
void GenerateInitialFeatureDistanceMapDatum(REGION_INFOS& region_infos,
        GTOR& grid2region, const Region& target_region, Datum& output);

/* ---------------------------------------------------------
 * Generate SLIC region information using vl-feat library.
 * ---------------------------------------------------------*/
void GenerateSlicRegions(Mat& image,
        REGION_INFOS& region_infos, GTOR& grid2region);

/* ---------------------------------------------------
 * Initialize low-level features of each SLIC region.
 * ---------------------------------------------------*/
void InitializeLowlevelFeatures(Mat& image, REGION_INFOS& region_infos);

/* ----------------------------------------------
 * Convert Image to Datum for VGG16 input
 * ----------------------------------------------*/
void GenerateImageDatum(Mat& _cv_img,
        vector<Datum>& img_vec, const int CROP_SIZE);

#endif
