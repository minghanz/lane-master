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
    VanPt(float alpha_w, float alpha_h);
    int initialVan(Mat color_img, Mat gray);
    void SteerFilter(Mat image, Mat& steer_resp, Mat& steer_angle_max, Mat& steer_resp_weight);
    void getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma);
    void GaborFilter(Mat image, Mat& gabor_resp_mag, Mat& gabor_resp_dir, Mat& gabor_weight);
    void GaborVote(Mat gabor_resp_dir, Mat gabor_weight, Mat& gabor_vote);
    void NMS(Mat matrix, Mat& matrix_nms);
    void LaneDirec(Mat steer_resp_mag, Mat edges, Mat& blur_edges);
    void RoughLaneDirec(Mat steer_resp_mag, Mat& mask_side, int direction);

    void DecideChnlThresh(Mat color_img, Mat image, Mat blur_edges); // use normalized one or ?
    void imageVan(Mat image);
    void recordBestVan(Line& left_lane, Line& right_lane);
    void recordHistVan(LaneImage& lane_find_image, Line& left_lane, Line& right_lane);
    void checkClearSeriesVan();
    void drawOn(Mat& newwarp, LaneMark& lane_mark);
    void renewWarp();
public:
    Point2f van_pt_ini;
    Point2f van_pt_img;
    Point2f van_pt_best;
    Point2f van_pt_avg;
    Point van_pt_best_int;
    Point2f van_pt_cali;
    #ifdef CALI_VAN
    float coef_pix_per_cm;
    float van_pt_cali_y;	
    float warp_pix_per_cm;
    int min_width_pixel_warp;
    #endif

    vector<Point2f> warp_src;
    vector<Point2f> warp_dst;
    vector<vector<Point> > warp_test_vec; // used in drawContour
    int y_bottom_warp;

	Mat per_mtx;
    Mat inv_per_mtx;
    float theta_w;	// yaw angle
    float theta_h;	// pitch angle
    const float ALPHA_W;
    const float ALPHA_H;
    bool first_sucs;                    // renewed by initialVan, fed to LaneImage, set hist_width if true
    bool sucs_before;           // renewed by initialVan, true if initialVan has ever succeeded, used to decide first_sucs only
    bool ini_flag; 

    bool consist;

    bool ini_success;   // used in GaborVote: deciding whether the vote has valid result
    vector<int> chnl_thresh;

private:
    vector<Point> series_van_pt;                    // renewed by renewSeriesVan, used by initialVan, cleared by fail_ini_count
	int pos_of_renew_van; // used in renewSeriesVan
    int fail_ini_count;
    const float VAN_TRACK_Y; // used in initialVan and renewSeriesVan
	
};

void localMaxima(const Mat& src, int num_nonz, int& min_loc, int hist_size);

void SteerFilter(Mat image, Mat& steer_resp, Mat sobel_angle, int ksize, double sigma);
void getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma);
