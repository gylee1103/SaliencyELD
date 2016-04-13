#ifndef COMMON_HPP
#define COMMON_HPP
#include <caffe/proto/caffe.pb.h>
using caffe::Datum;

enum {
	DIM_COLOR = 3,
	DIM_HIST_BIN = 8,
	DIM_LOCATION = 2,
	DIM_GABOR_BIN = 24,
	GRID_SIZE = 23,
	DIM_MODEL_FEED = 54,
	BGR_SPACE = 0,
	LAB_SPACE = 1,
	HSV_SPACE = 2,
	LABEL_SALIENT = 1,
	LABEL_NONSALIENT = 0,
	CVIMAGE_MAX = 255,
};

// Convienient structure
struct Pixel {
	int row;
	int col;
	int label;
	Pixel(int row_, int col_, int label_) {
		row = row_;
		col = col_;
		label = label_;
	}
};
typedef float Hist[DIM_COLOR][DIM_HIST_BIN];
typedef float Color[DIM_COLOR];
typedef int GTOR[GRID_SIZE][GRID_SIZE];

#endif
