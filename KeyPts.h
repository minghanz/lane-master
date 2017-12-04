#include "macro_define.h"

using namespace std;
using namespace cv;

extern float xm_per_pix;
extern float ym_per_pix;
extern int warp_col;
extern int warp_row;
extern Size img_size; // defined in main.cpp


class KeyPts
{
public: 
    vector<Point> key_left_2p, key_left_2n, key_right_2p, key_right_2n;
    vector<Point> key_left_2p_last, key_left_2n_last, key_right_2p_last, key_right_2n_last;
    vector<Point> key_left_2p_2now, key_left_2n_2now, key_right_2p_2now, key_right_2n_2now;
    Vec3f left_fit_key, right_fit_key;

    bool renew_flag;

public:
    void renew(vector<Point>& key_left_2p_new, vector<Point>& key_left_2n_new, vector<Point>& key_right_2p_new, vector<Point>& key_right_2n_new);
    void match();
    void fit();
};

void selectPt();