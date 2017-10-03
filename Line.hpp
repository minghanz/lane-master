#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor, Canny
#include <opencv2/calib3d.hpp>
#include <armadillo>

#include <string>
#include <iostream>
#include <vector>
#include <valarray>

#include <cstdlib>

#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;

class LaneImage;
class VanPt;
class LaneMark;

class Line
{
	public: 
	Line(float w_history = 0.7, bool sanity_check = true, bool sanity_parallel_check = true);
	void processNewRecord(VanPt& van_pt, LaneMark& lane_mark);
	float getDiff();
	void setSanityCheck(bool sanity_check, bool sanity_parallel_check);
	bool getSanityCheck();
	float getCurvature();
	float getDistanceToLane();
	void fitRealWorldCurve();
	void pushNewRecord(LaneImage& lane_find_image, int direction);
	
	bool detected;
	float best_x;
	vector<Vec3f> current_fit;
	Vec3f best_fit;
	Vec3f best_fit_cr;
	vector<float> radius_of_curvature;
	float best_radius_of_curvature;
	vector<float> line_base_pos;
	float best_line_base_pos;
	
	Vec3f diffs;
	vector<float> history_diff;
	float current_diff, mean_hist_diff;
	int pos_of_renew_diff;
	int base_fluctuation;
	float w_history;
	bool check, parallel_check;
	int fail_detect_count;
	
	float __w_current;
	
	vector<Vec2f> current_fit_2;
	Vec2f best_fit_2;
};
