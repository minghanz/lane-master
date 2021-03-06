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
#include <cmath>

#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

class LaneImage;
class Line;
class LaneMark;

class VanPt
{
public:
    VanPt(float alpha_w, float alpha_h, vector<float>& params);
    void initialVan(Mat color_img, Mat gray, Mat& warped_img, LaneMark& lane_mark);
    void SteerFilter(Mat image, Mat& steer_resp, Mat& steer_angle_max, Mat& steer_resp_weight);
    void getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma);
    void GaborFilter(Mat image, Mat& gabor_resp_mag, Mat& gabor_resp_dir, Mat& gabor_weight);
    bool GaborVote(Mat gabor_resp_dir, Mat gabor_weight, Mat& gabor_vote, Mat edges);
    void NMS(Mat matrix, Mat& matrix_nms);
    void LaneDirec(Mat steer_resp_mag, Mat edges, Mat& blur_edges);
    void RoughLaneDirec(Mat steer_resp_mag, Mat& mask_side, int direction);

    void DecideChnlThresh(Mat color_img, Mat image, Mat blur_edges); // use normalized one or ?

    void drawOn(Mat& newwarp, LaneMark& lane_mark);
    void renewWarp();
public:
    Point2f van_pt_ini;
    Point2f van_pt_img;
    Point2f van_pt_best;
    Point2f van_pt_avg;
    Point van_pt_best_int;
    Point2f van_pt_cali;

    /// not in individual VanPt
    bool cali_van;
    // #ifdef CALI_VAN

    float coef_pix_per_cm;  // constant
    float warp_pix_per_cm;  // constant
    float warp_real_width;  // constant
    int margin_side;
    float van_pt_cali_y;    // constant
    float min_length_pixel_warp;
    float min_width_pixel_warp; // constant
    // #endif
    ///////////////////////////////
    vector<Vec4i> lines_vote;

    vector<Point2f> warp_src;
    vector<Point2f> warp_dst;
    vector<vector<Point>> warp_test_vec; // used in drawContour
    int y_bottom_warp;
    int y_bottom_warp_max;

    Mat per_mtx;
    Mat inv_per_mtx;

    Mat vote_lines_img;

    /// not in individual VanPt
    Mat valid_lines_map;
    /////////////////////////////

    bool ini_flag;      // based on ini_success, but use history to show whether the van_pt_ini is in good quality
    bool first_sucs;
    bool sucs_before;
    int fail_ini_count;

    bool ini_success;   // indicating whether a van_pt_obsv is generated // used in GaborVote: deciding whether the vote has valid result

    #ifdef CANNY_VOTE
    Point2f van_pt_obsv;
    float max_weight_left;
    float max_weight_right;
    float confidence;
    float conf_weight, conf_mse, conf_dist;
    // KalmanFilter kalman;
    float conf_c_x, conf_gamma_x, conf_c_y, conf_gamma_y;
    float conf_c_x_max, conf_c_x_min;
    float conf_c_y_min, conf_c_y_max;
    float conf_gamma_e, conf_c_e;
    float line_gamma_dist, line_c_dist;
    float line_gamma_dist_max, line_gamma_dist_min;

    bool edgeVote(Mat image, Mat edges);
    int checkValidLine(Vec4i line);
    float getLineWeight(Vec4i line);
    float getConfidence(const vector<Point2f>& van_pt_candi, const vector<float>& van_pt_candi_w, 
        const vector<float>& valid_lines_w_left, const vector<float>& valid_lines_w_right); //, Point2f van_pt_obsv);
    void updateFlags();
    void updateTrackVar();
    #endif

    float theta_w;	// yaw angle
    float theta_h;	// pitch angle
    const float ALPHA_W;
    const float ALPHA_H;
    /// not in individual VanPt

    vector<int> chnl_thresh;
    ////////////////////////////
	
};

void localMaxima(const Mat& src, int num_nonz, int& min_loc, int hist_size);

void SteerFilter(Mat image, Mat& steer_resp, Mat sobel_angle, int ksize, double sigma);
void getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma);
