#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp

class LaneImage;


class KeyPts
{
public: 
    vector<Point> key_left_2p, key_left_2n, key_right_2p, key_right_2n;
    vector<Point> key_left_2p_last, key_left_2n_last, key_right_2p_last, key_right_2n_last;
    vector<Point> key_left_2p_2now, key_left_2n_2now, key_right_2p_2now, key_right_2n_2now;

    bool ini_p_left, ini_p_right, end_p_left, end_p_right;


    vector<Vec4f> markings_left, markings_right, markings_left_last, markings_right_last; // a vector of <x1, y1, x2, y2>
    Mat coord_left, coord_right, coord_left_last, coord_right_last;
    vector<int> match_to_idx_left, match_to_idx_right;
    Mat R_left, t_left, R_right, t_right;
    bool match_or_not_left, match_or_not_right;
    bool predicting_left, predicting_right;
    Mat lane_out_img_copy;

    Vec3f left_fit_key, right_fit_key;

    bool renew_flag, ini_flag;

public:
    // KeyPts();
    void renew(vector<Point>& key_left_2p_new, vector<Point>& key_left_2n_new, vector<Point>& key_right_2p_new, vector<Point>& key_right_2n_new, bool& ini_p_left_new, bool& ini_p_right_new, bool& end_p_left_new, bool& end_p_right_new);
    void renew(LaneImage& lane_find_image);
    void match(LaneImage& lane_find_image);
    void fit();
};

void selectPt(Mat& lane_window_side, Mat& lane_out_img_copy, vector<Point>& plot_pts_warp, vector<Vec4f>& markings);
void match_side(vector<Vec4f>& markings, vector<Vec4f>& markings_last, bool match_from_new, Mat& coord_new, Mat& coord_last, vector<int>& match_to_idx, bool& match_or_not );
void match_esti(Mat& coord, Mat& coord_last, Mat& R, Mat& t);
float match_eval(Vec4f& from, Vec4f& to, bool match_from_new);
void compli_last(vector<Vec4f>& markings, vector<Vec4f>& markings_last, bool match_from_new, vector<int>& match_to_idx, Mat& R, Mat& t, Mat& lane_out_img_copy, bool& predicting);
void revise_fit(Vec3f& lane_fit, vector<Vec4f>& markings, Mat& lane_window_out_img );