#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor, Canny
#include <opencv2/calib3d.hpp> // for calibration
#include <opencv2/ml/ml.hpp> // for machine learning
//#include <armadillo>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>

#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

class LaneImage;

class LearnModel
{
public:
    #ifdef DTREE
    Ptr<ml::DTrees> BGR_tree, HLS_tree;
    #endif
    #ifdef LREGG
    Ptr<ml::LogisticRegression> BGR_regg, HLS_regg; 
    #endif

	Mat BGR_sample; // 2000 for 1280*720, 500 for 400*500
	Mat HLS_sample;
    int n_samp;
    
	Mat BGR_resp;
    Mat HLS_resp;
    
    int samp_cyc;
public:
    void pushNewRecord(LaneImage & lane_find_image);
    LearnModel();
};

