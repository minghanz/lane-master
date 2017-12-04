#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

class VehDetc 
{
public:
    Rect pos;
    int conti_count;
    int count;
    bool conti_valid;
    bool renewed;
    float IOU;
    double confi;
    float lat_speed;
    vector<vector<Point> > side_poly;
public:
    VehDetc(Rect& pos_cur, float IOU_cur, double confi_cur);
    void posrenew(Rect& cur_rect, double confi_cur, Mat& per_mtx, Mat& inv_per_mtx);
    void negrenew();
    void estside(Rect& cur_rect, Mat& per_mtx, Mat& inv_per_mtx);
};

class VehMask
{
public:
    HOGDescriptor hog;
    vector<VehDetc> valid_detc;
    vector<VehDetc> track_detc;

    vector<Ptr<Tracker> > trackers;

    bool track_mode;

    Mat ori_veh_mask;
	Mat warp_veh_mask;
public:
    VehMask();
    void detectHOG(Mat& subimg, Mat& per_mtx, Mat& inv_per_mtx);
    void drawOn(Mat& newwarp);
    void indicateOnWarp(Mat& warped_raw_img);
    
};

