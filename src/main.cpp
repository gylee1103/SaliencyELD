#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/chrono.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include "common.hpp"
#include "region.hpp"
#include "extract_model_input.hpp"

namespace fs = boost::filesystem;
using std::vector;
using std::pair;
using std::string;
using cv::Mat;
using caffe::Net;
using caffe::Solver;
using caffe::Blob;
using caffe::MemoryDataLayer;

int main(int argc, char** argv) {
    FLAGS_minloglevel = 2;
    std::srand(3); // random seed for reproductivity(training)
    if (argc != 2) {
        printf("./program $test_dir\n");
        return -1;
    }
    const string test_image_dirname = argv[1];

    boost::chrono::system_clock::time_point time_start;

    /*--------------------------------------------------
     * Some Const Settings (e.g. model path, model blob names)
     * -------------------------------------------------*/
    const string VGG16_MODEL = 
        "../models/VGG16/VGG_ILSVRC_16_layers.caffemodel";
	static string VGG16_PROTO =
        "../models/VGG16/VGG_ILSVRC_16_layers_deploy_with_reduction.prototxt";
    const int VGG16_INPUT_SIZE = 224;
    const string VGG16_INPUT_LAYER_NAME = "image_input";
    const string VGG16_OUTPUT_FEATURE_NAME = "conv1_high";

	const string ELD_PROTO = "../models/ELD/deploy.prototxt";
	const string ELD_MODEL = "../models/ELD/eldmodel_iter_112192.caffemodel";
    const string ELD_INPUT_LOW_NAME = "lowlevel_db";
    const string ELD_INPUT_HIGH_NAME = "highlevel_db";
    const string ELD_SCORE_NAME = "score";

    /*---------------------------------------
     * Model initialize
     *--------------------------------------*/
    caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
    boost::shared_ptr<Net<float> > ELD_model;
    boost::shared_ptr<Net<float> > VGG16Net_model;
    ELD_model.reset(new Net<float>(ELD_PROTO, caffe::TEST));
    ELD_model->CopyTrainedLayersFrom(ELD_MODEL);

    // Concat 1x1 reduction layer to VGG
    VGG16Net_model.reset(new Net<float>(VGG16_PROTO, caffe::TEST));
    VGG16Net_model->CopyTrainedLayersFrom(VGG16_MODEL);
    VGG16Net_model->CopyTrainedLayersFrom(ELD_MODEL); // get 1x1 compress layer
     
    /*---------------------------------------
     * Collect the filenames in the test directory.
     *--------------------------------------*/
    printf("Collect filenames\n");
    fs::directory_iterator end_iter;
    vector<pair<string, string> > img_gt_filenames;

    fs::path test_image_dirpath(test_image_dirname), gtmask_dirpath;
    fs::directory_iterator imgdir_iter;
    for (imgdir_iter = fs::directory_iterator(test_image_dirpath);
            imgdir_iter != end_iter; imgdir_iter++) {
        string imgname = imgdir_iter->path().string();
        if (imgname.find("_ELD.png") != string::npos) continue; // Skip results
        string gtmask_name = ""; // not used in Test mode
        img_gt_filenames.push_back(std::make_pair(imgname, gtmask_name));
    }

    std::sort(img_gt_filenames.begin(), img_gt_filenames.end()); // convenient
    int total_image_number = img_gt_filenames.size();

    printf("Total file number : %d\n", total_image_number);

    /* ----------------------------------------
     * Loop each file.
     * ---------------------------------------*/
    
    for (int inum = 0; inum < total_image_number; inum++) {
        time_start = boost::chrono::system_clock::now(); // Check running time
        // Resize image. 
        string image_path = img_gt_filenames[inum].first;
        printf("Current image(%d) : %s\n", inum, image_path.c_str());
        Mat image = cv::imread(image_path, 1);
        // Generate region information
        GTOR grid2region; // Grid2Region map by region number
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                grid2region[i][j] = -1;
            }
        }
        REGION_INFOS region_infos; // key : region number
        GenerateSlicRegions(image, region_infos, grid2region);
        InitializeLowlevelFeatures(image, region_infos);

        /* ----------------------------------------
         * Generate High-level feature map
         * --------------------------------------*/
        vector<Datum> img_vec;
        GenerateImageDatum(image, img_vec, VGG16_INPUT_SIZE);
        const boost::shared_ptr<MemoryDataLayer<float> > img_input_layer =
                boost::static_pointer_cast<MemoryDataLayer<float> >(
                        VGG16Net_model->layer_by_name(VGG16_INPUT_LAYER_NAME));
        img_input_layer->set_batch_size(1);
        img_input_layer->AddDatumVector(img_vec);
        vector<Blob<float>*> empty_vec;
        VGG16Net_model->Forward(empty_vec);
        boost::shared_ptr<Blob<float> > pool3_blob = VGG16Net_model->
            blob_by_name(VGG16_OUTPUT_FEATURE_NAME);
        const boost::shared_ptr<Blob<float> >* blob_ptr_arr[1] = 
            {&pool3_blob}; // Can be extended further (multiple level features)
        Datum highlevel_datum_set[1];

        // Make VGG16 feature map input Datum
        for (int k = 0; k < 1; k ++) {
            // Blob to Datum
            const boost::shared_ptr<Blob<float> >& current_blob = 
                (*blob_ptr_arr[k]);
            highlevel_datum_set[k].set_channels(current_blob->channels());
            highlevel_datum_set[k].set_height(current_blob->height());
            highlevel_datum_set[k].set_width(current_blob->width());
            highlevel_datum_set[k].clear_data();
            highlevel_datum_set[k].clear_float_data();
            for (int c = 0; c < current_blob->channels(); c++) {
                float* feature_blob_data = current_blob->mutable_cpu_data() +
                    current_blob->offset(0, c);
                for (int h = 0; h < current_blob->height(); h++) {
                    for (int w = 0; w < current_blob->width(); w++) {
                        highlevel_datum_set[k].add_float_data(
                            feature_blob_data[h * current_blob->width() + w]);
                    }
                }
            }
        }

        /* ----------------------------------------------------
         * Process our model
         * input : Initial Feature Distance Map
         * input : VGG16 conv5_3 compressed by one 1x1 layer
         * ---------------------------------------------------- */
        Mat result;
        cvtColor(image, result, CV_BGR2GRAY);
        vector<Datum> all_lowlevel_datums(region_infos.size());
        vector<Datum> all_highlevel_datums(region_infos.size(),
                                           highlevel_datum_set[0]);

        // Generate Initial Feature Distance Map of all query regions
#pragma omp parallel
#pragma omp single
{
        for (std::map<int, Region>::const_iterator it = 
                std::begin(region_infos); it != std::end(region_infos); it++) {
#pragma omp task firstprivate(it)
            {
                int index = std::distance(region_infos.cbegin(), it);
                Datum lowlevel_datum;
                GenerateInitialFeatureDistanceMapDatum(region_infos,
                        grid2region, it->second, all_lowlevel_datums[index]);
            }
        }
}
        // Query all regions concurrently
        const boost::shared_ptr<MemoryDataLayer<float> > low_input_layer =
            boost::static_pointer_cast<MemoryDataLayer<float> >(
                    ELD_model->layer_by_name(ELD_INPUT_LOW_NAME));
        const boost::shared_ptr<MemoryDataLayer<float> > high_input_layer =
            boost::static_pointer_cast<MemoryDataLayer<float> >(
                    ELD_model->layer_by_name(ELD_INPUT_HIGH_NAME));

        low_input_layer->set_batch_size(all_lowlevel_datums.size());
        low_input_layer->AddDatumVector(all_lowlevel_datums);
        high_input_layer->set_batch_size(all_lowlevel_datums.size());
        high_input_layer->AddDatumVector(all_highlevel_datums);
        empty_vec.clear();
        ELD_model->Forward(empty_vec);

        const boost::shared_ptr<Blob<float>> score_blob = 
                ELD_model->blob_by_name(ELD_SCORE_NAME);
        // Save result to outputdir
        int idx = 0;
        for (auto regpair : region_infos) {
            float* score_blob_data = score_blob->mutable_cpu_data() +
                score_blob->offset(idx++);
            float saliency_score = score_blob_data[1];
            int format_score = saliency_score * 255;
            for (auto pixel : regpair.second.get_pixels()) {
                result.at<uchar>(pixel.row, pixel.col) =
                    static_cast<unsigned char>(format_score);
            }
        }
        boost::chrono::duration<double> sec = 
            boost::chrono::system_clock::now() - time_start;
        printf("time : %f\n", sec.count());
        fs::path savepath(image_path);
        string outname = savepath.string();
        // note: This simply replace ".jpg" to "_ELD.png"
        // please change this if you use different file extension.
        outname.replace(outname.end()-4, outname.end(), "_ELD.png");
        cv::imwrite(outname, result);
    }
    return 0;
}
