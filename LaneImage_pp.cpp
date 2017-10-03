#include "LaneImage.hpp"
#include "Line.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

////// changing to caltech data, need to change rowRange (4 places), total_pix and y_bottom_max!!!



void illuComp(Mat& raw_img, Mat& gray, float& illu_comp)
{
	int mode = 2; // 1: top 5%; 2: 20%-80% sample 10%
	
	/// histogram for reference white
	int hist_size = 256;
	float range_bgrls[] = {0, 256};
	const float* range_5 = {range_bgrls};
	int hist_size_h = 180;
	float range_h[] = {0, 180};
	const float* range_1 = {range_h};
	bool uniform = true;
	bool accumulate = false;
	
	Mat pixhis_gray;
	Mat height_mask_2(gray.size(), CV_8UC1, Scalar(255));
	#ifndef HIGH_BOT
	height_mask_2.rowRange(0, gray.rows*2/3) = 0;
	#else
	height_mask_2.rowRange(0, gray.rows/2) = 0;
	height_mask_2.rowRange(gray.rows*7/10, gray.rows) = 0;  // for caltech data
	#endif
	
	calcHist( &gray, 1, 0, height_mask_2, pixhis_gray, 1, &hist_size, &range_5, uniform, accumulate );
	float accu_num_pix = 0;
	float accu_graylevel = 0;
	#ifndef HIGH_BOT
	float total_pix = gray.rows * gray.cols/3 ; 
	#else
	float total_pix = gray.rows * gray.cols/10*2 ; // for caltech data
	#endif
	
	if (mode == 1)
	{
		for (int i = 0; i < hist_size; i++)
		{
			accu_num_pix += pixhis_gray.at<float>(hist_size - 1 - i);
			accu_graylevel += pixhis_gray.at<float>(hist_size - 1 - i) * (hist_size - 1 - i);
			if (accu_num_pix >= 0.05*total_pix)
				break;
		}
		float base_white = accu_graylevel/accu_num_pix;
		cout << "base_white value: " << base_white << endl;
		illu_comp = 100/base_white;
	}
	else
	{
		bool start = false, ending = false;
		int limit_low = 0, limit_high = 0;
		float effective_num_pix = 0;
		for (int i = 0; i < hist_size; i++)
		{
			if ( start == false && accu_num_pix >= 0.2*total_pix )
			{
				limit_low = i;
				start = true;
				effective_num_pix = accu_num_pix;
				cout << "effective_num_pix 1: " << effective_num_pix << endl;
			}
			if ( accu_num_pix >= 0.8*total_pix && ending == false )
			{
				limit_high = i;
				ending = true;
				cout << "effective_num_pix 2: " << effective_num_pix << endl;
				effective_num_pix = accu_num_pix - effective_num_pix;
				cout << "accu_num_pix: " << accu_num_pix << endl;
				cout << "effective_num_pix 3: " << effective_num_pix << endl;
				break;
			}
			accu_num_pix += pixhis_gray.at<float>(i);
			if (start == true && ending == false)
				accu_graylevel += pixhis_gray.at<float>(i) * i;
		}
		float avg_gray = accu_graylevel / effective_num_pix;
		//cout << pixhis_gray << endl;
		cout << "total_pix: " << total_pix << ", accu_num_pix: " << accu_num_pix << endl;
		cout << accu_graylevel << " " << effective_num_pix << endl;
		cout << "avg_gray: " << avg_gray<< endl;
		illu_comp = 80 / avg_gray;
	}
	
	#ifndef NDEBUG_IN
	imshow("image before compen", raw_img);
	#endif
	gray = gray*illu_comp;
	raw_img = raw_img*illu_comp;
	#ifndef NDEBUG_IN
	imshow("image after compen", raw_img);
	waitKey(0);
	#endif
	return;
}


float LaneImage::get_curvature(int side)
{
	float y_eval_loc = 0*ym_per_pix; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + 0.1; // changed to counting from closer side
	
	valarray<float> curverad(10);
	if (side == 1) // left
		curverad = pow(( 1 + pow(( 2*__left_fit_cr[2]*y_eval*ym_per_pix +__left_fit_cr[1]), 2)),1.5)/abs(2*__left_fit_cr[2]);
	else
		curverad = pow(( 1 + pow(( 2*__right_fit_cr[2]*y_eval*ym_per_pix +__right_fit_cr[1]), 2)),1.5)/abs(2*__right_fit_cr[2]);
	float mean_curverad = curverad.sum()/curverad.size();
	
	return mean_curverad;
}

float LaneImage::get_distance_to_lane(int side)
{
	float y_eval_loc =0*ym_per_pix; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + 0.1; // changed to counting from closer side
	float veh_loc = __col/2*xm_per_pix;
	
	valarray<float> x0(10);
	if (side == 1) // left
		x0 = __left_fit_cr[2]*y_eval*y_eval + __left_fit_cr[1]*y_eval + __left_fit_cr[0];
	else
		x0 = __right_fit_cr[2]*y_eval*y_eval + __right_fit_cr[1]*y_eval + __right_fit_cr[0];
	float distance = x0.sum()/x0.size() - veh_loc;
	
	#ifndef NDEBUG
	cout << "veh loc: " << veh_loc << endl;
	cout << "lane loc: " << x0.sum()/x0.size() << endl;
	#endif
	
	return distance;
}

Vec3f LaneImage::get_lane_fit(int side)
{
	if (side == 1)
		return __left_fit;
	else
		return __right_fit;
}

Mat LaneImage::get_warp_filter_image()
{
	return __warped_filter_image;
}

Mat LaneImage::get_calibrated_image()
{
	return __calibrate_image;
}
