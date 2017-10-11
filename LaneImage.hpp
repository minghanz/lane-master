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

class VanPt;
class LearnModel;
class LaneMark;



void cameraCalibration(vector<vector<Point3f> >& obj_pts, vector<vector<Point2f> >& img_pts, Size& image_size, int ny=6, int nx=9);
void import(int argc, char** argv, string& file_name, Mat& image, VideoCapture& reader, VideoWriter& writer, int msc[], int& ispic, int& video_length);

class LaneImage
{
	public:
	//// #ifdef DTREE
	//// LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp,
	//// 	Vec3f left_fit = Vec3f(0, 0, 0), Vec3f right_fit = Vec3f(0, 0, 0), Vec3f avg_hist_left_fit = Vec3f(0, 0, 0), Vec3f avg_hist_right_fit = Vec3f(0, 0, 0), vector<int> chnl_thresh = vector<int>(6, 0), Ptr<ml::DTrees> BGR_tree = Ptr<ml::DTrees>(),Ptr<ml::DTrees> HLS_tree = Ptr<ml::DTrees>(), Mat dist_coeff = Mat(), Mat cam_mtx = Mat() );
	//// #endif
	//// #ifdef LREGG
	//// LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp,
	//// 	Vec3f left_fit = Vec3f(0, 0, 0), Vec3f right_fit = Vec3f(0, 0, 0), Vec3f avg_hist_left_fit = Vec3f(0, 0, 0), Vec3f avg_hist_right_fit = Vec3f(0, 0, 0), vector<int> chnl_thresh = vector<int>(6, 0), Ptr<ml::LogisticRegression> BGR_regg = Ptr<ml::LogisticRegression>(),Ptr<ml::LogisticRegression> HLS_regg = Ptr<ml::LogisticRegression>(), Mat dist_coeff = Mat(), Mat cam_mtx = Mat() );
	//// #endif
	LaneImage(Mat image, VanPt& van_pt, LaneMark& lane_mark, LearnModel& learn_model, Mat cam_mtx, Mat dist_coeff, float nframe);

	void __calibration();
	void __laneBase(int& hist_width);
	void __ROIInds(int half_width, valarray<float>& nonzx, valarray<float>& nonzy, valarray<bool>& left_lane_inds, valarray<bool>& right_lane_inds );
	// void __reshapeSub(int half_width, const Mat& warp_reshape, const Mat& warp_reshape_HLS, Mat& warp_reshape_sub, Mat& warp_reshape_sub_HLS, vector<Point>& sub_pts  );

	void __imageFilter();
	void __warp();
	void trainmodel(Mat& warped_filter_image_U, valarray<float>& nonzx, valarray<float>& nonzy, valarray<bool>& left_lane_inds, valarray<bool>& right_lane_inds);
	void __fitLaneMovingWindow(int& hist_width, bool& last_all_white);
	void __makeUpFilter(bool left, Mat& warped_filter_image_U, vector<Point>& nonz_loc, valarray<float>& nonzx, valarray<float>& nonzy, int& hist_width, valarray<float>& leftx, valarray<float>& lefty, valarray<float>& rightx, valarray<float>& righty);
	float __getDiff(Vec3f& cur_fit, Vec3f& hist_fit);
	float __getCurveDiff(Vec3f& cur_fit, Vec3f& hist_fit); // consistent with function getLaneWidthWarp

	void __laneSanityCheck(int hist_width);
	float get_curvature(int side);
	float get_distance_to_lane(int side);
	Vec3f get_lane_fit(int side);
	Mat get_warp_filter_image();
	Mat get_calibrated_image();
	void get_vanishing_point(Mat inv_per_mtx);
	//// void processVan(Line& left_lane, Line& right_lane, Point2f& last_van_pt);
	//// void processVan(Vec2f left_fit_2, Vec2f right_fit_2);
	
	Mat __raw_image;
	int __row, __col;
	Mat __calibration_dist;
	Mat __calibration_mtx;
	
	int __leftx_base, __rightx_base;
	
	int __sobel_kernel_size;
	Vec2f __r_thresh;
	Vec2f __g_thresh;
	Vec2f __b_thresh;
	Vec2f __h_thresh;
	Vec2f __l_thresh;
	Vec2f __s_thresh;
	Vec2f __abs_x_thresh_pre;
	Vec2f __abs_x_thresh;
	Vec2f __abs_y_thresh;
	Vec2f __mag_thresh;
	Vec2f __dir_thresh;
	Mat __filter_binary; // not used
	
	Mat __sobelx;
	Mat __filter_binary_x_p, __filter_binary_x_n; // for makeUpFilter
	bool __left_nolane, __right_nolane;
	
	int __samp_cyc;		// __fitLaneMovingWindow (decide the part of samples to renew)
	float __nframe; 	// __fitLaneMovingWindow (decide whether to retrain dtree)
	bool __train_or_not;
	Mat __BGR_sample, __HLS_sample;
	Mat __BGR_resp, __HLS_resp;
	#ifdef DTREE
	Ptr<ml::DTrees> __BGR_tree, __HLS_tree;
	#endif
	#ifdef LREGG
	Ptr<ml::LogisticRegression> __BGR_regg, __HLS_regg;
	#endif
	
	Mat __transform_matrix;
	
	int __window_number;
	float __window_half_width;		// for __imageFilter(dilate and move), __ROIInds(decide window width)
	int __window_min_pixel;
	
	bool __initial_frame;
	Vec3f __last_left_fit, __last_right_fit;
	Vec3f __left_fit, __right_fit;
	Vec3f __left_fit_cr, __right_fit_cr;
	valarray<float> __leftx, __lefty, __rightx, __righty;
	Mat __lane_window_out_img;
	
	Vec3f __avg_hist_left_fit, __avg_hist_right_fit;
	float __left_dist_to_hist, __right_dist_to_hist; // to find the more consist lane
	float __left_curve_dist_to_hist, __right_curve_dist_to_hist;
	
	bool __first_sucs; // for finding lane base
	int __min_width_warp;
	
	float __mean_dist; // the renewing of window_width is based on refined lane
	
	float __dif_dist, __dif_curve, __time_curve, __bot_width; // for check
	bool __parallel_check, __width_check;
	// bool __van_consist;
	
	Vec2f __left_fit_2, __right_fit_2; // in warped perspective
	Vec2f __left_fit_2_img, __right_fit_2_img; // in original perspective
	Point __van_pt;
	Point __best_van_pt;
	
	Mat __calibrate_image;
	Mat __warped_raw_image;
	Mat __warped_filter_image;
	Mat __warped_reshape;
	Mat __warped_reshape_HLS;
	
};
#ifdef DTREE
void trainTree(Ptr<ml::DTrees> tree, Mat& sample, Mat& resp, Mat& testdata, size_t orignl_row, string field);
#endif
#ifdef LREGG
void trainRegg(Ptr<ml::LogisticRegression> regg, Mat& sample, Mat& resp, Mat& testdata, size_t orignl_row, string field);
#endif
void colorThresh(const Mat& image, Mat& binary_output, Vec2f thresh, int layer, string colormap);
void sobelAbsThresh(const Mat& sobel, Mat& binary_output, Vec2f thresh);
//void sobelMagThresh(const Mat& sobelx, const Mat& sobely, Mat& binary_output, Vec2f thresh);
void sobelDirThresh(const Mat& sobelx, const Mat& sobely, Mat& binary_output, Vec2f thresh);

void illuComp(Mat& raw_img, Mat& gray, float& illu_comp);

void extractPeaks(const Mat& src, const int min_peak_dist, const float min_height_diff, const float min_height, vector<int>& max_loc, vector<float>& max_val);


string x2str(int num);
string x2str(float num);
string x2str(double num);
string x2str(bool num);
