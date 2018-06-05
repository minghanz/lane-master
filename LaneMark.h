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

class LaneImage; // forward declaration
class Line;
class VanPt;

class LaneMark
{
public:
	Vec3f left_fit_img;	// results from LaneImage
    Vec3f right_fit_img;
    Vec3f left_fit_best;	// results from Line's best result
	Vec3f right_fit_best;
	float left_fit_w;	// weight from Line
	float right_fit_w;
	
	int window_half_width;    // 12 // fed to LaneImage(__window_half_width), and Line (window_half_width). Decided by getLaneWidthWarp
	
	int hist_width;             // fed to LaneImage, renewed by recordHistFit
	Vec3f avg_hist_left_fit;       // fed to LaneImage, renewed by recordHistFit
	Vec3f avg_hist_right_fit;
	int pos_of_renew_fit_left;          // used by recordHistFit
	int pos_of_renew_fit_right;

	vector<Vec3f> hist_fit_left;            // used by recordHistFit
	vector<Vec3f> hist_fit_right;

	bool new_result;
	bool initial_frame;
	bool last_all_white;

	bool split; // indicating whether a branch of lane is coming
	bool new_branch_found;
	int split_recover_count;
	int branch_grow_count;
	bool branch_at_left;
	float k_pitch, b_pitch;
public:
	LaneMark();
	void recordImgFit(LaneImage& lane_find_image);
	void recordBestFit(Line& left_lane, Line& right_lane, VanPt& van_pt);
	void recordHistFit();
	void drawOn(Mat& newwarp, VanPt& van_pt, LaneImage& lane_find_image, ofstream& pointfile);
};

void recordHistFit_(vector<Vec3f>& hist_fit, Vec3f& avg_hist_fit, Vec3f& new_fit, int& pos_of_renew_fit, bool initial_frame); // have delay of three
float getLaneWidthWarp(Vec3f left_fit, Vec3f right_fit);