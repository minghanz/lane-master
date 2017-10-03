#include "LearnModel.h"
#include "LaneImage.hpp"

LearnModel::LearnModel()
{
    BGR_sample = Mat(warp_col * warp_row / 400,3, CV_32FC1); // 2000 for 1280*720, 500 for 400*500
	HLS_sample = Mat(warp_col * warp_row / 400,3, CV_32FC1);
    n_samp = warp_col * warp_row / 400 / 5;

    samp_cyc = 0;

    #ifdef DTREE
	BGR_resp = Mat( n_samp*5, 1, CV_32SC1);
    HLS_resp = Mat( n_samp*5, 1, CV_32SC1);
    #endif

    #ifdef LREGG    
	BGR_resp = Mat( n_samp*5, 1, CV_32FC1);
    HLS_resp = Mat( n_samp*5, 1, CV_32FC1);
    #endif
}

void LearnModel::pushNewRecord(LaneImage& lane_find_image)
{
    #ifdef DTREE
    BGR_tree = lane_find_image.__BGR_tree;
    HLS_tree = lane_find_image.__HLS_tree;
    #endif
    #ifdef LREGG
    BGR_regg = lane_find_image.__BGR_regg;
    HLS_regg = lane_find_image.__HLS_regg;
    #endif
    BGR_sample = lane_find_image.__BGR_sample;
    HLS_sample = lane_find_image.__HLS_sample;
    BGR_resp = lane_find_image.__BGR_resp;
    HLS_resp = lane_find_image.__HLS_resp;

    if ( lane_find_image.__train_or_not )
    {
        samp_cyc = (samp_cyc + 1) % 5;
    }
}