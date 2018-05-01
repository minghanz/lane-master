#include "LaneImage.hpp"
#include <cstdio>

#include "VanPt.h"
#include "LaneMark.h"
#include "LearnModel.h"
#include "VehMask.h"
#include "KeyPts.h"

using namespace std;
using namespace cv;

//float ym_per_pix = 40./720.;
//float xm_per_pix = 3.7/600.;
float ym_per_pix = 40./500.;
float xm_per_pix = 3.7/200.;

int warp_col = 300; // 400
int warp_row = 400;	// 500

//// #ifdef DTREE
//// LaneImage::LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp, 
//// 	Vec3f left_fit, Vec3f right_fit, Vec3f avg_hist_left_fit, Vec3f avg_hist_right_fit, vector<int> chnl_thresh, Ptr<ml::DTrees> BGR_tree, Ptr<ml::DTrees> HLS_tree, Mat dist_coeff, Mat cam_mtx )
//// #endif
//// #ifdef LREGG
//// LaneImage::LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp, 
//// 	Vec3f left_fit, Vec3f right_fit, Vec3f avg_hist_left_fit, Vec3f avg_hist_right_fit, vector<int> chnl_thresh, Ptr<ml::LogisticRegression> BGR_regg, Ptr<ml::LogisticRegression> HLS_regg, Mat dist_coeff, Mat cam_mtx )
//// #endif
LaneImage::LaneImage(Mat image, VanPt& van_pt, LaneMark& lane_mark, LearnModel& learn_model, VehMask& veh_masker, KeyPts& key_pts, float nframe)
{
	// __raw_image = image;
	// __row = image.rows;
	// __col = image.cols;
	// __calibration_dist = dist_coeff;
	// __calibration_mtx = cam_mtx;
	// __calibration();

	// __calibrate_image = image; 

	__left_fit = Vec3f(0, 0, 0);
	__right_fit = Vec3f(0, 0, 0);
	__left_fit_cr = Vec3f(0, 0, 0);
	__right_fit_cr = Vec3f(0, 0, 0);
	__k_pitch = lane_mark.k_pitch;
	__b_pitch = lane_mark.b_pitch;

	float y0 = -__b_pitch/__k_pitch;	// record last frame's k and b for finding valid pixels, but decay to half
	float y1 = 1/__k_pitch + y0;
	y0 = 2*y0;
	__k_pitch = -1/(y0-y1);
	__b_pitch = y0/(y0-y1);

	__split = lane_mark.split;
	__new_branch_found = false;
	__split_recover_count = lane_mark.split_recover_count;
	__branch_grow_count = lane_mark.branch_grow_count;
	__branch_at_left = lane_mark.branch_at_left;

	if (!van_pt.ini_flag)
	{
		cout << "Current frame is not processed due to failed initialization. " << endl;
	}
	else
	{
		__warped_raw_image = image;

		///////////////////////// color/gradient filter parameters
		__sobel_kernel_size = max(warp_col/80, 5 );		//__sobel_kernel_size = 15;
		vector<int> chnl_thresh = van_pt.chnl_thresh;
		if (chnl_thresh[0] == 0)
		{
		__r_thresh = Vec2f(90, 255); // 2
		__g_thresh = Vec2f(60, 255); // +
		__b_thresh = Vec2f(70, 255); // w
		__h_thresh = Vec2f(0, 25); // y
		__l_thresh = Vec2f(55, 255); // +
		__s_thresh = Vec2f(85, 255); // y
		__abs_x_thresh = Vec2f(10, 100);
		__abs_y_thresh = Vec2f(10, 100);
		__mag_thresh = Vec2f(20, 200);
		__dir_thresh = Vec2f(50, 200); // [0-pi/2] -> [0-255]
		//__r_thresh = Vec2f(170, 255);
		//__g_thresh = Vec2f(190, 255);
		//__b_thresh = Vec2f(190, 255);
		//__h_thresh = Vec2f(18, 100);
		//__l_thresh = Vec2f(100, 180);
		//__s_thresh = Vec2f(55, 255);
		//__abs_x_thresh = Vec2f(5, 20);
		//__abs_y_thresh = Vec2f(5, 20);
		//__mag_thresh = Vec2f(10, 40);
		//__dir_thresh = Vec2f(0.7, 1.0);
		}
		else
		{
			__r_thresh = Vec2f(chnl_thresh[2], 255); // 2
			__g_thresh = Vec2f(chnl_thresh[1], 255); // +
			__b_thresh = Vec2f(chnl_thresh[0], 255); // w
			__h_thresh = Vec2f(0, chnl_thresh[3]); // y
			__l_thresh = Vec2f(chnl_thresh[4], 255); // +
			__s_thresh = Vec2f(chnl_thresh[5], 255); // y
			__abs_x_thresh_pre = Vec2f(10, 255); // (10,100)
			__abs_x_thresh = Vec2f(20, 255); // (10,100) (20, 255)
			__abs_y_thresh = Vec2f(0, 30); // (10, 50) (0, 20) not used
			__mag_thresh = Vec2f(50, 255); // (20, 100) not used
			__dir_thresh = Vec2f(0, 100); // [0-pi/2] -> [0-255] (50, 200)
			#ifndef NDEBUG_CL
			cout << "Color thresh: " << __b_thresh[0] << " " << __g_thresh[0] << " " << __r_thresh[0] << " " << __h_thresh[1] << " " << __l_thresh[0] << " " << __s_thresh[0] << " " << endl;
			cout << "Adaptive color threshold used. " << endl;
			#endif
		}
		///////////////////////// decision tree parameters
		#ifdef DTREE
		if ( learn_model.BGR_tree.empty() )
		{
			__BGR_tree = ml::DTrees::create();
			cout << "new BGR tree." << endl;
		}
		else
		{
			__BGR_tree = learn_model.BGR_tree;
		}
		if ( learn_model.HLS_tree.empty() )
		{
			__HLS_tree = ml::DTrees::create();
			cout << "new HLS tree." << endl;
		}
		else
		{
			__HLS_tree = learn_model.HLS_tree;
			
		}
		#endif
		#ifdef LREGG
		if ( learn_model.BGR_regg.empty() )
		{
			__BGR_regg = ml::LogisticRegression::create();
			cout << "new BGR regg." << endl;
		}
		else
		{
			__BGR_regg = learn_model.BGR_regg;
		}
		if ( learn_model.HLS_regg.empty() )
		{
			__HLS_regg = ml::LogisticRegression::create();
			cout << "new HLS regg." << endl;
		}
		else
		{
			__HLS_regg = learn_model.HLS_regg;
			
		}
		#endif
		
		__BGR_sample = learn_model.BGR_sample;
		__HLS_sample = learn_model.HLS_sample;
		__BGR_resp = learn_model.BGR_resp;
		__HLS_resp = learn_model.HLS_resp;
		__samp_cyc = learn_model.samp_cyc;
		__nframe = nframe;
		__train_or_not = false;
		
		__left_nolane = false;
		__right_nolane = false;

		///////////////////////// fitting parameters
		__window_number = 10;
		//__window_half_width = warp_col/12;
		__window_half_width = lane_mark.window_half_width;
		__window_min_pixel = warp_row/__window_number*__window_half_width*2/10; // 1%.  400/10*25*2/100 = 20  400/10*25*2/10 = 200
		#ifndef NDEBUG_FT
		cout << "min pixel: " << __window_min_pixel << endl;
		#endif
		
		__initial_frame = ! lane_mark.new_result;
		__last_left_fit = lane_mark.left_fit_best;
		__last_right_fit = lane_mark.right_fit_best;
		
		__left_dist_to_hist = 0;
		__right_dist_to_hist = 0;
		__left_curve_dist_to_hist = 0;
		__right_curve_dist_to_hist = 0;
		__avg_hist_left_fit = lane_mark.avg_hist_left_fit;
		__avg_hist_right_fit = lane_mark.avg_hist_right_fit;
		
		__first_sucs = van_pt.first_sucs;
		// #ifdef CALI_VAN
		__min_width_warp = van_pt.min_width_pixel_warp;
		__min_marking_length = van_pt.min_length_pixel_warp;
		// #else
		// __min_width_warp = warp_col/6;
		// #endif

		///////////////////////////// start to find lanes
		clock_t t_last = clock();
		// __transform_matrix = van_pt.per_mtx;
		// __warp();
		
		//__filter_binary = Mat::ones(__row, __col, CV_32F);
		__warped_filter_image = Mat::ones(warp_row, warp_col, CV_32F);
		__imageFilter( veh_masker.warp_veh_mask );

		// __findKeyPt( veh_masker ); // based on built-in API of feature point extraction
		
		clock_t t_now = clock();
		cout << "Image filtered, using " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
		t_last = t_now;
		
		__lane_window_out_img = Mat(warp_row, warp_col, CV_8UC3, Scalar(0, 0, 0));
		__fitLaneMovingWindow(lane_mark.hist_width, lane_mark.last_all_white, veh_masker);

		// __findKeyCustom(veh_masker, key_pts); // feature points in old format

		t_now = clock();
		cout << "Image fitted, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
		t_last = t_now;
		
		// if (__left_fit != Vec3f(0, 0, 0) && __right_fit != Vec3f(0, 0, 0))
		// {
		// 	//__getLaneWidthWarp();
		// 	get_vanishing_point(van_pt.inv_per_mtx);
		// }
		
		// t_now = clock();
		// cout << "New vanishing point found, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
		// t_last = t_now;
	}
	
	
	
	
}


// void LaneImage::__calibration()		// undistort is done out of LaneImage now
// {
// 	/// generate an undistorted image (no perspective transforms)
// 	if (__calibration_mtx.empty())
// 		__calibrate_image = __raw_image;
// 	else
// 		undistort(__raw_image, __calibrate_image, __calibration_mtx, __calibration_dist);
// 	return;
// }

// void LaneImage::__warp()				// warp is done out of LaneImage now
// {
// 	/// generate warped binary image or colorful image
// 	warpPerspective(__calibrate_image, __warped_raw_image, __transform_matrix, Size(warp_col, warp_row), INTER_NEAREST);
// 	//warpPerspective(__filter_binary, __warped_filter_image, __transform_matrix, Size(__col, __row), INTER_NEAREST);
// 	#ifndef NDEBUG
// 	imshow("warped_raw", __warped_raw_image);
// 	#endif
// 	return;
// }

void LaneImage::__findKeyCustom(VehMask& veh_masker, KeyPts& key_pts)
{
	if ( __left_fit!= Vec3f(0, 0, 0) && __right_fit != Vec3f(0, 0, 0) )
	{
		Mat lane_window_out_r(warp_row, warp_col, CV_8UC1);
		Mat lane_window_out_l(warp_row, warp_col, CV_8UC1);
		Mat out[] = {lane_window_out_r, lane_window_out_l};
		int from_to[] = {2, 0, 0, 1};
		mixChannels(&__lane_window_out_img, 1, out, 2, from_to, 2);

		Mat lane_out_img_copy;
		__lane_window_out_img.copyTo(lane_out_img_copy);

		cout << "entering key custom" << endl;


		vector<Point> key_left_2p, key_left_2n;
		bool ini_p_left, end_p_left;
		selectPt(lane_window_out_l, lane_out_img_copy, __plot_pts_lr_warp[0], key_left_2p, key_left_2n, ini_p_left, end_p_left);
		cout << "finish key custom 1 " << endl;

		vector<Point> key_right_2p, key_right_2n;
		bool ini_p_right, end_p_right;
		selectPt(lane_window_out_r, lane_out_img_copy, __plot_pts_lr_warp[1], key_right_2p, key_right_2n, ini_p_right, end_p_right);
		cout << "finish key custom 2 " << endl;

		key_pts.renew( key_left_2p, key_left_2n, key_right_2p, key_right_2n, ini_p_left, ini_p_right, end_p_left, end_p_right);

		imshow("key_custom", lane_out_img_copy);
		
	}
}
void selectPt(Mat& lane_window_side, Mat& lane_out_img_copy, vector<Point>& plot_pts_warp, vector<Point>& key_2p, vector<Point>& key_2n, bool& ini_p, bool& end_p)
{
	int up_thresh = 4, low_thresh = 1, dist_thresh = 10;
	vector<int> near_nonznum(plot_pts_warp.size(), 0);

	if (plot_pts_warp[0].x >= 0 && plot_pts_warp[0].x < warp_col )
		near_nonznum[0] = countNonZero(lane_window_side(Range(plot_pts_warp[0].y, plot_pts_warp[0].y + 1), Range(max(plot_pts_warp[0].x - 5, 0), min(plot_pts_warp[0].x + 5, warp_col))));

	cout << "near_nonznum init" << endl;
	
	Point cache_point(0, 0);
	float cache_ctnonz = 0;
	// bool cur_p = lane_window_side.at<uchar>(plot_pts_warp[0][0]) > 0;
	bool cur_p = near_nonznum[0] >= up_thresh;
	ini_p = cur_p;	// to indicate whether the first keypt is 2n (ini_p = true) or 2p (ini_p = false)
	for (int i = 1; i < plot_pts_warp.size(); i++)
	{
		if (plot_pts_warp[i].x >= 0 && plot_pts_warp[i].x < warp_col )
			near_nonznum[i] = countNonZero(lane_window_side(Range(plot_pts_warp[i].y, plot_pts_warp[i].y + 1), Range(max(plot_pts_warp[i].x - 5, 0), min(plot_pts_warp[i].x + 5, warp_col))));

		cache_ctnonz += near_nonznum[i];

		if ( !cur_p && near_nonznum[i] >= up_thresh) // (!cur_p || key_2n.size() + key_2p.size() == 0) // lane_window_side.at<uchar>(plot_pts_warp[i]) > 0
		{
			bool dist_ok = true;
			// if (key_2n.size() > 0 && plot_pts_warp[i].y - key_2n.back().y < dist_thresh)
			// {
			// 	dist_ok = false;
			// }
			if (cache_point.y != 0 && plot_pts_warp[i].y - cache_point.y < dist_thresh)
			{
				dist_ok = false;
			}
			
			if (dist_ok)
			{
				if (cache_point.y != 0)
				{
					if (cache_ctnonz/(plot_pts_warp[i].y-cache_point.y) < up_thresh) // up_thresh
					{
						key_2n.push_back(cache_point);
						cache_point = plot_pts_warp[i];
						cache_ctnonz = 0;
						circle(lane_out_img_copy, key_2n.back(), 3, Scalar(0, 255, 0), -1);
						cur_p = true;
					}
					else
					{
						i = cache_point.y;
						cache_point = Point(0,0);
						cur_p = true;
					}
				}
				else
				{
					cache_point = plot_pts_warp[i];
					cache_ctnonz = 0;
					cur_p = true;
				}
			}
			// key_2p.push_back(plot_pts_warp[i]);
			// circle(lane_out_img_copy, key_2p.back(), 3, Scalar(0, 255, 0), 1);
			// cur_p = true;
		}
		else if (cur_p && near_nonznum[i] <= low_thresh) // (cur_p || key_2n.size() + key_2p.size() == 0) // lane_window_side.at<uchar>(plot_pts_warp[i]) == 0
		{
			bool dist_ok = true;
			// if (key_2p.size() > 0 && plot_pts_warp[i].y - key_2p.back().y < dist_thresh)
			// {
			// 	dist_ok = false;
			// }
			if (cache_point.y != 0 && plot_pts_warp[i].y - cache_point.y < dist_thresh)
			{
				dist_ok = false;
			}
			
			if (dist_ok)
			{
				if (cache_point.y != 0)
				{
					if (cache_ctnonz/(plot_pts_warp[i].y-cache_point.y) > low_thresh) // low_thresh
					{
						key_2p.push_back(cache_point);
						cache_point = plot_pts_warp[i];
						cache_ctnonz = 0;
						circle(lane_out_img_copy, key_2p.back(), 3, Scalar(0, 255, 0), 1);
						cur_p = false;
					}
					else
					{
						i = cache_point.y;
						cache_point = Point(0,0);
						cur_p = false;
					}
				}
				else
				{
					cache_point = plot_pts_warp[i];
					cache_ctnonz = 0;
					cur_p = false;
				}
			}
			// if (dist_ok)
			// {
			// 	key_2n.push_back(plot_pts_warp[i]);
			// 	circle(lane_out_img_copy, key_2n.back(), 3, Scalar(0, 255, 0), 1);
			// 	cur_p = false;
			// }
		}
	}
	end_p = cur_p;
	if (cache_point != Point(0, 0) )
	{
		if ( !cur_p )
		{
			key_2n.push_back(cache_point);
			circle(lane_out_img_copy, key_2n.back(), 3, Scalar(0, 255, 0), -1);
		}
		else
		{
			key_2p.push_back(cache_point);
			circle(lane_out_img_copy, key_2p.back(), 3, Scalar(0, 255, 0), 1);
		}
	}
}

void LaneImage::__findKeyPt( VehMask& veh_masker )
{
	// xfeatures2d:: SIFT SURF BriefDescriptorExtractor
	// Ptr<xfeatures2d::FastFeatureDetector> sift_detector = xfeatures2d::FastFeatureDetector::create();
	//  FastFeatureDetector AgastFeatureDetector GFTTDetector SimpleBlobDetector AKAZE* BRISK*(slow) KAZE* (MSER) ORB*
	Ptr<ORB> sift_detector = ORB::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;

	Mat out_image;
	sift_detector->detectAndCompute(__warped_raw_image, ~veh_masker.warp_veh_mask, keypoints, descriptors, false );
	// sift_detector->detect(__warped_raw_image, keypoints, ~warp_veh_mask);
	drawKeypoints( __warped_raw_image, keypoints, out_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	if (veh_masker.valid_detc.size() >= 1)
    {
        vector<Mat> channels;
        split(out_image, channels);
        channels[2] = veh_masker.warp_veh_mask + channels[2] + 0;
        merge(channels, out_image);
    }
	imshow("keypoint", out_image);
	// waitKey(0);
}

void LaneImage::__imageFilter(Mat& warp_veh_mask)
{
	/// filter the warped image
	Mat gray;
	cvtColor(__warped_raw_image, gray, COLOR_BGR2GRAY); // BGR
	Mat warp_mask = gray == 0;
	
	GaussianBlur(gray, gray, Size(5,5), 0 );
	
	Mat dilate_kernel = getStructuringElement(MORPH_RECT, Size(5, 5) );		// prepare for excluding the black zone induced by warping effect
	dilate(warp_mask, warp_mask, dilate_kernel );
	#ifndef NDEBUG_CL
	imshow("warp_mask", warp_mask);
	#endif
	
	
	
	Mat binary_output(warp_row, warp_col, CV_32FC1, Scalar(1));
	Mat filter_binary;
	//RGB filter
	/* // based on the unwarped calibrated image
	colorThresh(__calibrate_image, filter_binary, __b_thresh, 0, "rgb");
	Mat binary_output_white = binary_output.mul(filter_binary);
	colorThresh(__calibrate_image, filter_binary, __g_thresh, 1, "rgb");
	binary_output_white = binary_output_white.mul(filter_binary);
	colorThresh(__calibrate_image, filter_binary, __r_thresh, 2, "rgb");
	binary_output_white = binary_output_white.mul(filter_binary);
	threshold(binary_output_white, binary_output_white, 0.2, 1, THRESH_TOZERO);
	//HLS filter
	colorThresh(__calibrate_image, filter_binary, __h_thresh, 0, "hls");
	Mat binary_output_yellow = binary_output.mul(filter_binary);
	colorThresh(__calibrate_image, filter_binary, __l_thresh, 1, "hls");
	binary_output_yellow = binary_output_yellow.mul(filter_binary);
	colorThresh(__calibrate_image, filter_binary, __s_thresh, 2, "hls");
	binary_output_yellow = binary_output_yellow.mul(filter_binary);
	threshold(binary_output_yellow, binary_output_yellow, 0.2, 1, THRESH_TOZERO);
	
	Mat binary_output_color;
	addWeighted(binary_output_white, 1, binary_output_yellow, 1, 0, binary_output_color);
	double max_val;
	minMaxLoc(binary_output_color, NULL, &max_val, NULL, NULL);
	binary_output_color = binary_output_color*(1/max_val);
	threshold(binary_output_color, binary_output_color, 0.4, 1, THRESH_TOZERO);
	*/
	time_t t_temp1 = clock();
	
	double max_val;
	/*
	#ifdef DTREE
	Mat binary_output_white(warp_row, warp_col, CV_32FC1, Scalar(0)), binary_output_yellow(warp_row, warp_col, CV_32FC1, Scalar(0));
	#endif
	#ifdef LREGG
	Mat binary_output_white(warp_row, warp_col, CV_32SC1, Scalar(0)), binary_output_yellow(warp_row, warp_col, CV_32SC1, Scalar(0));
	#endif
	*/
	
	Mat binary_output_white(warp_row, warp_col, CV_32FC1, Scalar(0)), binary_output_yellow(warp_row, warp_col, CV_32FC1, Scalar(0));
	Mat binary_output_color(warp_row, warp_col, CV_32FC1, Scalar(0));

	// need to reshape no matter whether this is initial_frame, because the training of tree will use it
	__warped_reshape = __warped_raw_image.reshape(1, warp_row*warp_col);	
	__warped_reshape.convertTo(__warped_reshape, CV_32FC1);
	Mat warp_HLS;
	cvtColor(__warped_raw_image, warp_HLS, COLOR_BGR2HLS);
	__warped_reshape_HLS = warp_HLS.reshape(1, warp_row*warp_col);
	__warped_reshape_HLS.convertTo(__warped_reshape_HLS, CV_32FC1);
	
	#ifndef NDEBUG_CL
	cout << "reshape finished. " << endl;
	#endif
	
	if (__initial_frame || __BGR_tree->getRoots().empty() || __HLS_tree->getRoots().empty() ) // __nframe == 0
	{
		time_t t_temp2 = clock();
		/*
		binary_output_white.convertTo(binary_output_white,CV_32FC1);
		binary_output_yellow.convertTo(binary_output_yellow,CV_32FC1);
		*/
		colorThresh(__warped_raw_image, filter_binary, __b_thresh, 0);
		binary_output_white = binary_output.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked b-", filter_binary); // no need to namedWindow
		#endif
		colorThresh(__warped_raw_image, filter_binary, __g_thresh, 1);
		binary_output_white = binary_output_white.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked g-", filter_binary);
		#endif
		colorThresh(__warped_raw_image, filter_binary, __r_thresh, 2);
		binary_output_white = binary_output_white.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked r-", filter_binary);
		#endif
		colorThresh(warp_HLS, filter_binary, __h_thresh, 0);
		binary_output_white = binary_output_white.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked h-", filter_binary);
		#endif
		colorThresh(warp_HLS, filter_binary, __l_thresh, 1);
		binary_output_white = binary_output_white.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked l-", filter_binary);
		#endif
		colorThresh(warp_HLS, filter_binary, __s_thresh, 2);
		binary_output_white = binary_output_white.mul(filter_binary);
		#ifndef NDEBUG_CL
		imshow("masked s-", filter_binary);
		#endif
		threshold(binary_output_white, binary_output_white, 0.06, 1, THRESH_TOZERO);
		
		binary_output_white.setTo(0, warp_mask);
		
		minMaxLoc(binary_output_white, NULL, &max_val, NULL, NULL);
		binary_output_color = binary_output_white*(1/max_val);
		
		time_t t_temp3 = clock();
		cout << "Reshaped: " << to_string(((float)(t_temp2 - t_temp1))/CLOCKS_PER_SEC) << "s. Thresholded: ";
				cout << to_string(((float)(t_temp3 - t_temp2))/CLOCKS_PER_SEC) <<"s. Total: ";
				cout << to_string(((float)(t_temp3 - t_temp1))/CLOCKS_PER_SEC) << endl;
		
		#ifndef NDEBUG
		imshow("color binary", binary_output_color);
		//waitKey(0);
		#endif
	}
	else
	{
		time_t t_temp2 = clock();
		/*
		Mat warped_reshape_sub, warped_reshape_sub_HLS;
		vector<Point> sub_pts;
		//__reshapeSub((int)__window_half_width*2, __warped_reshape, __warped_reshape_HLS, warped_reshape_sub, warped_reshape_sub_HLS, sub_pts, binary_output_white, binary_output_yellow);
		__reshapeSub((int)__window_half_width*2, __warped_reshape, __warped_reshape_HLS, warped_reshape_sub, warped_reshape_sub_HLS, sub_pts);
		*/
		
		#ifdef DTREE
		__BGR_tree->predict(__warped_reshape, binary_output_white);
		__HLS_tree->predict(__warped_reshape_HLS, binary_output_yellow);
		#endif
		#ifdef LREGG
		__BGR_regg->predict(__warped_reshape, binary_output_white);
		__HLS_regg->predict(__warped_reshape_HLS, binary_output_yellow);
		binary_output_white.convertTo(binary_output_white, CV_32FC1);
		binary_output_yellow.convertTo(binary_output_yellow, CV_32FC1);
		#endif
		
		/*
		Mat sub_resp_BGR, sub_resp_HLS;
		#ifdef DTREE
		__BGR_tree->predict(warped_reshape_sub, sub_resp_BGR);
		__HLS_tree->predict(warped_reshape_sub_HLS, sub_resp_HLS);
		#endif
		#ifdef LREGG
		__BGR_regg->predict(warped_reshape_sub, sub_resp_BGR);
		__HLS_regg->predict(warped_reshape_sub_HLS, sub_resp_HLS);
		sub_resp_BGR.convertTo(sub_resp_BGR, CV_32FC1);
		sub_resp_HLS.convertTo(sub_resp_HLS, CV_32FC1);
		#endif
		cout << "size of sub_resp: " << sub_resp_BGR.size() << " " << sub_resp_HLS.size()  << endl;
		cout << "size of reshape_sub: " << warped_reshape_sub.size() << " " << warped_reshape_sub_HLS.size() << endl;
 		int n_sub = warped_reshape_sub.rows;
		for (int i = 0; i < n_sub; i++)
		{
			binary_output_white.at<float>(sub_pts[i]) = sub_resp_BGR.at<float>(i);
			binary_output_yellow.at<float>(sub_pts[i]) = sub_resp_HLS.at<float>(i);
		}
		*/
		
		time_t t_temp3 = clock();
		
		binary_output_white = binary_output_white.reshape(1, warp_row);
		binary_output_yellow = binary_output_yellow.reshape(1, warp_row);
		
		binary_output_color = (binary_output_white + binary_output_yellow)*0.5; // two possible values: 0.5 or 1 
		
		// double max_val_bi;
		// minMaxLoc(binary_output_color, NULL, &max_val_bi, NULL, NULL);
		// cout << "max of binary_output_color: " << max_val_bi << endl;
		
		binary_output_color.setTo(0, warp_mask);
		
		time_t t_temp4 = clock();
		cout << "Reshaped: " << to_string(((float)(t_temp2 - t_temp1))/CLOCKS_PER_SEC) << "s. Predicted: ";
				cout << to_string(((float)(t_temp3 - t_temp2))/CLOCKS_PER_SEC) <<"s. Reshaped: ";
				cout << to_string(((float)(t_temp4 - t_temp3))/CLOCKS_PER_SEC) <<"s. Total: ";
				cout << to_string(((float)(t_temp4 - t_temp1))/CLOCKS_PER_SEC) << endl;
		
		#ifndef NDEBUG
		// cout << binary_output_white.size() << endl;
		// cout << binary_output_yellow.size() << endl;
		cout << "reshape new color filter. " << endl;
		imshow("color binary", binary_output_color);
		//waitKey(0);
		#endif
		#ifndef NDEBUG_CL
		imshow("BGR binary", binary_output_white);
		imshow("HLS binary", binary_output_yellow);
		//waitKey(0);
		#endif
	}
	
	
	
	time_t t_temp5 = clock();
	/// gradient filter
	Mat sobelx, sobely;
	Sobel(gray, sobelx, CV_32F, 1, 0, __sobel_kernel_size );
	Sobel(gray, sobely, CV_32F, 0, 1, __sobel_kernel_size );
	
	sobelx.copyTo(__sobelx); // for __makeUpFilter
	
	Mat sobel_xp = sobelx > 0, sobel_xn = sobelx < 0;

	Mat filter_binary_x_p, filter_binary_x_n; //, __filter_binary_dir;
	sobelDirThresh(sobelx, sobely, __filter_binary_dir, __dir_thresh);

	Mat edges(gray.size(), gray.type(), Scalar(0));
	Canny(gray.rowRange(0, warp_row/2), edges.rowRange(0, warp_row/2), 20, 40, 3);
	Canny(gray.rowRange(warp_row/2, warp_row), edges.rowRange(warp_row/2, warp_row), 30, 60, 3);

	filter_binary_x_p = edges & __filter_binary_dir & sobel_xp;
	filter_binary_x_n = edges & __filter_binary_dir & sobel_xn;

	int dilate_width = max( (int)round(__window_half_width*0.4),4);
	int move_dist = max( (int)round(__window_half_width*0.2),2);
	Mat expand_kernel = getStructuringElement(MORPH_RECT, Size(dilate_width, 1));

	// {
	// 	Mat sobel_xp, sobel_xn;
	// 	// threshold(sobelx, sobel_xp, 0, 1, THRESH_TOZERO);
	// 	sobel_xn = -sobelx;
	// 	// threshold(sobel_xn, sobel_xn, 0, 1, THRESH_TOZERO);

	// 	// sobelAbsThresh(sobel_xp, __filter_binary_x_p, __abs_x_thresh_pre);   // relative threshold is not good!!!
	// 	// sobelAbsThresh(sobel_xn, __filter_binary_x_n, __abs_x_thresh_pre);
	// 	__filter_binary_x_p = sobelx > 300;
	// 	__filter_binary_x_n = sobel_xn > 300;
		
	// 	__filter_binary_x_p = __filter_binary_x_p & __filter_binary_dir;
	// 	__filter_binary_x_n = __filter_binary_x_n & __filter_binary_dir;   // for __makeUpFilter

	// 	Mat filter_binary_p_2left(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
	// 	Mat filter_binary_n_2left(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
	// 	Mat filter_binary_p_2right(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
	// 	Mat filter_binary_n_2right(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
	// 	filter_binary_p_2right.colRange(move_dist, warp_col) = __filter_binary_x_p.colRange(0, warp_col - move_dist) + 0;
	// 	filter_binary_p_2left.colRange(0, warp_col - move_dist) = __filter_binary_x_p.colRange(move_dist, warp_col) + 0;
	// 	filter_binary_n_2right.colRange(move_dist, warp_col) = __filter_binary_x_n.colRange(0, warp_col - move_dist) + 0;
	// 	filter_binary_n_2left.colRange(0, warp_col - move_dist) = __filter_binary_x_n.colRange(move_dist, warp_col) + 0;

	// 	dilate(filter_binary_p_2right, filter_binary_p_2right, expand_kernel);
	// 	dilate(filter_binary_p_2left, filter_binary_p_2left, expand_kernel);
	// 	dilate(filter_binary_n_2right, filter_binary_n_2right, expand_kernel);
	// 	dilate(filter_binary_n_2left, filter_binary_n_2left, expand_kernel);
		
	// }

	time_t t_temp6 = clock();
	
	
	#ifndef NDEBUG_GR
	cout << "dilate width: " << dilate_width << endl; // /5
	// imshow("filter_binary_p", filter_binary_x_p);
	// imshow("filter_binary_n", filter_binary_x_n);
	#endif
	Mat filter_binary_p_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	Mat filter_binary_n_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	filter_binary_p_move.colRange(move_dist, warp_col) = filter_binary_x_p.colRange(0, warp_col - move_dist) + 0;
	filter_binary_n_move.colRange(0, warp_col - move_dist) = filter_binary_x_n.colRange(move_dist, warp_col) + 0;
	#ifndef NDEBUG_GR
	// imshow("filter_binary_p_dilate 1", filter_binary_p_move);
	// imshow("filter_binary_n_dilate 1", filter_binary_n_move);
	cout << "move dist: " << move_dist << endl;
	#endif
	
	dilate(filter_binary_p_move, filter_binary_p_move, expand_kernel);
	dilate(filter_binary_n_move, filter_binary_n_move, expand_kernel);
	#ifndef NDEBUG_GR
	cout << "expand_kernel size: " << expand_kernel.size() << endl;
	imshow("filter_binary_p_dilate", filter_binary_p_move);
	imshow("filter_binary_n_dilate", filter_binary_n_move);
	#endif
	/*Mat*/ __binary_output_gradient = filter_binary_p_move & filter_binary_n_move;
	
	#ifndef NDEBUG_GR
	imshow("__binary_output_gradient 1", __binary_output_gradient);
	#endif
	
	Mat dilate_kernel_2 = getStructuringElement(MORPH_RECT, Size(1, 7));
	dilate(__binary_output_gradient, __binary_output_gradient, dilate_kernel_2);

	Mat labels, stats, centroids;
	int total_marks =  connectedComponentsWithStats(__binary_output_gradient, labels, stats, centroids);
	for (int i = 0; i < total_marks; i++)
	{
		if ( (stats.at<int>(i, CC_STAT_AREA) / stats.at<int>(i, CC_STAT_HEIGHT) > 20 && stats.at<int>(i, CC_STAT_AREA) > 100 ) ||
		 (stats.at<int>(i, CC_STAT_HEIGHT) < __min_marking_length && stats.at<int>(i, CC_STAT_TOP)!= 0 && stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT) < warp_row-1) )
		{
			for (int j = stats.at<int>(i, CC_STAT_TOP); j < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT); j++)
			{
				for (int k = stats.at<int>(i, CC_STAT_LEFT); k < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH); k++)
				{
					if (labels.at<int>(j, k)== i)
					{
						__binary_output_gradient.at<uchar>(j, k) = 0;
					}
				}
			}
		}
		else if (stats.at<int>(i, CC_STAT_HEIGHT) < 2*__min_marking_length && stats.at<int>(i, CC_STAT_WIDTH) >= __min_marking_length ) // 80 30
		{
			bool bulk = false;
			for (int j = stats.at<int>(i, CC_STAT_TOP); j < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT); j++)
			{
				int start_line = -1, end_line = -1;
				for (int k = stats.at<int>(i, CC_STAT_LEFT); k < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH); k++)
				{
					if (labels.at<int>(j, k)== i)
					{
						start_line = k;
						break;
					}
				}

				if (start_line + 30 > stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH) )
					continue;
				
				for (int k = stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH) -1; k >= stats.at<int>(i, CC_STAT_LEFT); k--)
				{
					if (labels.at<int>(j, k)== i)
					{
						end_line = k;
						break;
					}
				}

				if (end_line - start_line >= 30)
				{
					bulk = true;
					break;
				}
			}
			if (bulk)
			{
				for (int j = stats.at<int>(i, CC_STAT_TOP); j < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT); j++)
				{
					for (int k = stats.at<int>(i, CC_STAT_LEFT); k < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH); k++)
					{
						if (labels.at<int>(j, k)== i)
						{
							__binary_output_gradient.at<uchar>(j, k) = 0;
						}
					}
				}
			}
		}
	}

	#ifndef NDEBUG_GR
	imshow("__binary_output_gradient 2", __binary_output_gradient);
	#endif

	// sobelx.setTo(0, ~__binary_output_gradient);
	// Mat abs_sobelx = abs(sobelx);
	__binary_output_gradient.convertTo(__binary_output_gradient, binary_output_color.type(), 1.0/255.0 ); // 1.0/(255.0*2)





	// ////////////////////////// first version, before 1/9/2018
	// Mat sobelx, sobely;
	// Sobel(gray, sobelx, CV_32F, 1, 0, __sobel_kernel_size );
	// Sobel(gray, sobely, CV_32F, 0, 1, __sobel_kernel_size );
	
	// sobelx.copyTo(__sobelx); // for __makeUpFilter
	
	// Mat sobel_xp, sobel_xn;
	// threshold(sobelx, sobel_xp, 0, 1, THRESH_TOZERO);
	// sobel_xn = -sobelx;
	// threshold(sobel_xn, sobel_xn, 0, 1, THRESH_TOZERO);
	
	// Mat filter_binary_x_p, filter_binary_x_n, __filter_binary_dir;
	// sobelAbsThresh(sobel_xp, filter_binary_x_p, __abs_x_thresh_pre);
	// sobelAbsThresh(sobel_xn, filter_binary_x_n, __abs_x_thresh_pre);
	// sobelDirThresh(sobelx, sobely, __filter_binary_dir, __dir_thresh);
	
	// time_t t_temp6 = clock();
	
	// filter_binary_x_p = filter_binary_x_p & __filter_binary_dir;
	// filter_binary_x_n = filter_binary_x_n & __filter_binary_dir;
	
	// __filter_binary_x_n = filter_binary_x_n;  // for __makeUpFilter
	// __filter_binary_x_p = filter_binary_x_p;
	
	// int dilate_width = max( (int)round(__window_half_width*0.4),4);
	// int move_dist = max( (int)round(__window_half_width*0.2),2);
	
	// #ifndef NDEBUG_GR
	// cout << "dilate width: " << dilate_width << endl; // /5
	// imshow("filter_binary_p", filter_binary_x_p);
	// imshow("filter_binary_n", filter_binary_x_n);
	// #endif
	// Mat filter_binary_p_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	// Mat filter_binary_n_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	// filter_binary_p_move.colRange(move_dist, warp_col) = filter_binary_x_p.colRange(0, warp_col - move_dist) + 0;
	// filter_binary_n_move.colRange(0, warp_col - move_dist) = filter_binary_x_n.colRange(move_dist, warp_col) + 0;
	// #ifndef NDEBUG_GR
	// imshow("filter_binary_p_dilate 1", filter_binary_p_move);
	// imshow("filter_binary_n_dilate 1", filter_binary_n_move);
	// cout << "move dist: " << move_dist << endl;
	// #endif
	// Mat expand_kernel = getStructuringElement(MORPH_RECT, Size(dilate_width, 1));
	// dilate(filter_binary_p_move, filter_binary_p_move, expand_kernel);
	// dilate(filter_binary_n_move, filter_binary_n_move, expand_kernel);
	// #ifndef NDEBUG_GR
	// cout << "expand_kernel size: " << expand_kernel.size() << endl;
	// imshow("filter_binary_p_dilate", filter_binary_p_move);
	// imshow("filter_binary_n_dilate", filter_binary_n_move);
	// #endif
	// /*Mat*/ __binary_output_gradient = filter_binary_p_move & filter_binary_n_move;
	
	// #ifndef NDEBUG_GR
	// imshow("__binary_output_gradient 1", __binary_output_gradient);
	// #endif
	
	// sobelx.setTo(0, ~__binary_output_gradient);
	// Mat abs_sobelx = abs(sobelx);
	// __binary_output_gradient.convertTo(__binary_output_gradient, binary_output_color.type(), 0 ); // 1.0/(255.0*2)
	
	// Mat sobel_xtop(warp_row, warp_col, CV_32FC1, Scalar(0));
	// Mat sobel_xbot(warp_row, warp_col, CV_32FC1, Scalar(0));
	// sobel_xtop.rowRange(0, warp_row/2) = abs_sobelx.rowRange(0, warp_row/2) + 0;
	// sobel_xbot.rowRange(warp_row/2, warp_row) = abs_sobelx.rowRange(warp_row/2, warp_row) + 0;
	
	// Mat filter_binary_x_top, filter_binary_x_bot;
	// sobelAbsThresh(sobel_xtop, filter_binary_x_top, __abs_x_thresh);
	// sobelAbsThresh(sobel_xbot, filter_binary_x_bot, __abs_x_thresh);
	// __binary_output_gradient.setTo(1, filter_binary_x_top | filter_binary_x_bot);
	
	// /////////////////////////////////////////////////////////



	#ifndef NDEBUG_GR
	imshow("__binary_output_gradient 2", __binary_output_gradient); // two possible values: 0.5 or 1
	#endif
	
	
	#ifndef NDEBUG
	imshow("gradient binary", __binary_output_gradient);
	//waitKey(0);
	#endif
	
	//Mat mask = binary_output_color <= 0;
	//__binary_output_gradient.setTo(0, mask);
	
	time_t t_temp7 = clock();
	
	/// combine together
	Mat binary_output_final;
	addWeighted(binary_output_color, 1, __binary_output_gradient, 1, 0, binary_output_final); // four possible values: 0.25, 0.5, 0.75, 1
	
	minMaxLoc(binary_output_final, NULL, &max_val, NULL, NULL);
	binary_output_final = binary_output_final*(1/max_val);
	threshold(binary_output_final, __warped_filter_image, 0.4, 1, THRESH_TOZERO); // filter out the lowest possibility

	__warped_filter_image.setTo(0, warp_veh_mask );

	
	#ifndef NDEBUG
	imshow("overall binary", __warped_filter_image);
	//waitKey(0);
	#endif
	
	time_t t_temp8 = clock();
	cout << "Gradient processed: " << to_string(((float)(t_temp6 - t_temp5))/CLOCKS_PER_SEC) << "s. Thresholded: ";
		cout << to_string(((float)(t_temp7 - t_temp6))/CLOCKS_PER_SEC) <<"s. Combined: ";
			cout << to_string(((float)(t_temp8 - t_temp7))/CLOCKS_PER_SEC) <<"s. Total: ";
			cout << to_string(((float)(t_temp8 - t_temp5))/CLOCKS_PER_SEC) <<"s. " << endl;
	return;
}

void colorThresh(const Mat& image, Mat& binary_output, Vec2f thresh, int layer)
{
	// Mat image_cur;
	// if (colormap == "hls")
	// 	cvtColor(image, image_cur, COLOR_BGR2HLS); // RGB or BGR???
	// else
	// 	image_cur = image; // not copied
		
	Mat color_channel(warp_row, warp_col, CV_8UC1);
	int from_to[2] = {layer, 0};
	mixChannels(image, color_channel, from_to, 1);
	
	binary_output = Mat::ones(color_channel.size(), CV_32FC1);
	Mat mask = (color_channel <= thresh[0]) | (color_channel >= thresh[1]);
	binary_output.setTo(0.5, mask);
	return;
}


void sobelAbsThresh(const Mat& abs_sobel, Mat& binary_output, Vec2f thresh)
{
	// Mat abs_sobel;
	// threshold(abs_sobel, abs_sobel, 0, 1, THRESH_TOZERO);
	
	//Mat abs_sobel = abs(sobel);
	
	//normalize(abs_sobel, abs_sobel, 0, 255, NORM_MINMAX );
	double max_val;
	minMaxLoc(abs_sobel, NULL, &max_val, NULL, NULL);
	float thresh_l = thresh[0]/255*max_val;
	float thresh_h = thresh[1]/255*max_val;
	
	binary_output = (abs_sobel >= thresh_l) & (abs_sobel <= thresh_h);
	
	#ifndef NDEBUG_GR
	imshow("sobel abs", binary_output);
	waitKey(0);
	#endif
	
	return;
}


void sobelDirThresh(const Mat& sobelx, const Mat& sobely, Mat& binary_output, Vec2f thresh)
{
	Mat sobel_dir;
	phase(abs(sobelx), abs(sobely), sobel_dir); // not in degree
	//normalize(sobel_dir, sobel_dir, 0, 255, NORM_MINMAX ); // normalized to 0-255
	double max_val, min_val;
	minMaxLoc(sobel_dir, &min_val, &max_val, NULL, NULL);
	float thresh_l = thresh[0]/255*(max_val - min_val) + min_val;
	float thresh_h = thresh[1]/255*(max_val - min_val) + min_val;
	
	binary_output = (sobel_dir >= thresh_l) & (sobel_dir <= thresh_h );
	
	#ifndef NDEBUG_GR
	imshow("sobel dir", binary_output);
	#endif
	
	return;
}


void LaneImage::__laneBase(int& hist_width)
{
	if ( __initial_frame ) // originally only left
	{

		__split = false;
		__new_branch_found = false;
		__split_recover_count = 0;
		__branch_grow_count = 0;
		__branch_at_left = false;
		__k_pitch = 1e-10;
		__b_pitch = 1;


		/// histogram to find the base for fitting
		Mat histogram(1, warp_col, CV_32FC1, Scalar(0));
		//float* line_his = histogram.ptr<float>();
		for (int i = warp_row/2; i < warp_row; i++ )
		{
			//const float* line = __warped_filter_image.ptr<float>(i);
			//for (int j = 0; j < warp_col; j++)
				//line_his[j] += line[j];
			histogram += __warped_filter_image.row(i);
		}
		
		int midpoint = warp_col/2;
		double max_histogram_l, max_histogram_r;
		minMaxIdx(histogram.colRange(0, midpoint), NULL, &max_histogram_l, NULL, NULL);
		minMaxIdx(histogram.colRange(midpoint, warp_col), NULL, &max_histogram_r, NULL, NULL);
		
		/// finding peaks and prefer peaks that do not have other peaks in between
		int min_peak_dist = warp_col / 20;
		float min_height_diff_l = max_histogram_l/3, min_height_diff_r = max_histogram_r/3; // 5
		float min_height_l = min_height_diff_l, min_height_r = min_height_diff_r;
		vector<int> max_loc_l, max_loc_r;
		vector<float> max_val_l, max_val_r;
		extractPeaks(histogram.colRange(0, midpoint), min_peak_dist, min_height_diff_l, min_height_l, max_loc_l, max_val_l);
		extractPeaks(histogram.colRange(midpoint, warp_col), min_peak_dist, min_height_diff_r, min_height_r, max_loc_r, max_val_r);


		// Mat hist_peaks(1, max_val_l.size() + max_val_r.size(), CV_32FC1, Scalar(0)); // embed width constraints into finding peaks
		// for (int i = 0; i < max_loc_l.size(); i++)
		// {
		// 		//float sub_max = 0;
		// 		float cur_peak = max_val_l[i];
		// 		for (int j = i; j < max_loc_l.size(); j++)
		// 		{
		// 			if (j > i) // && max_loc[j] < warp_col/2 && max_val[j] > sub_max)
		// 			{
		// 				//sub_max = max_val[j];
		// 				cur_peak = cur_peak - 2*max_val_l[j];
		// 			}
		// 		}
		// 		hist_peaks.at<float>( i ) = cur_peak; // max_val[i] - sub_max;
		// }
		// for (int i = 0; i < max_loc_r.size(); i++)
		// {
		// 		//float sub_max = 0;
		// 		float cur_peak = max_val_r[i];
		// 		for (int j = i; j >= 0; j--)
		// 		{
		// 			if (j < i  ) // && max_loc[j] > warp_col/2 && max_val[j] > sub_max)
		// 			{
		// 				//sub_max = max_val[j];
		// 				cur_peak = cur_peak - 2*max_val_r[j];
		// 			}
		// 		}
		// 		hist_peaks.at<float>( i + max_loc_l.size() ) = cur_peak; // max_val[i] - sub_max;
		// }
		// Mat hist_peaks_pair(1, max_val_l.size() * max_val_r.size(), CV_32FC1, Scalar(0));
		// for (int i = 0; i < max_loc_l.size(); i++)
		// {
		// 	for (int j = 0; j < max_loc_r.size(); j++)
		// 	{
		// 		hist_peaks_pair.at<float>(i * max_loc_r.size() + j) = (hist_peaks.at<float>(i) + hist_peaks.at<float>(j + max_loc_l.size())) * (midpoint + max_loc_r[j] - max_loc_l[i] >= __min_width_warp);
		// 	}
		// }
		// int base_pair[2];
		// minMaxIdx(hist_peaks_pair, NULL, NULL, NULL, base_pair); //histogram.colRange(0, midpoint)
		// int peak_pair = base_pair[1];
		// int peak_r_idx = peak_pair % max_loc_r.size();
		// int peak_l_idx = peak_pair / max_loc_r.size();
		
		// __leftx_base = max_loc_l[peak_l_idx];
		// __rightx_base = max_loc_r[peak_r_idx] + midpoint;
		// hist_width = __rightx_base - __leftx_base;
		
		Mat hist_peaks(1, warp_col, CV_32FC1, Scalar(0));
		for (int i = 0; i < max_loc_l.size(); i++)
		{
				//float sub_max = 0;
				float cur_peak = max_val_l[i];
				for (int j = i; j < max_loc_l.size(); j++)
				{
					if (j > i) // && max_loc[j] < warp_col/2 && max_val[j] > sub_max)
					{
						//sub_max = max_val[j];
						cur_peak = cur_peak - 2*max_val_l[j];
					}
				}
				hist_peaks.at<float>( max_loc_l[i] ) = cur_peak; // max_val[i] - sub_max;
		}
		for (int i = 0; i < max_loc_r.size(); i++)
		{
				//float sub_max = 0;
				float cur_peak = max_val_r[i];
				for (int j = i; j >= 0; j--)
				{
					if (j < i  ) // && max_loc[j] > warp_col/2 && max_val[j] > sub_max)
					{
						//sub_max = max_val[j];
						cur_peak = cur_peak - 2*max_val_r[j];
					}
				}
				hist_peaks.at<float>( max_loc_r[i] + midpoint ) = cur_peak; // max_val[i] - sub_max;
		}
		
		/// extract peaks in histogram
		
		//if (__first_sucs) // only the very first time initialization succeeds
		//{
			int leftx_base_p[2];
			int rightx_base_p[2];
			/*
			Mat mask_hist(1, warp_col, CV_8UC1, Scalar(0));
			Mat inv_mask_hist(1, warp_col, CV_8UC1, Scalar(0));
			uchar* line_mask = mask_hist.ptr();
			uchar* inv_line_mask = inv_mask_hist.ptr();
			for (int i = 0; i < midpoint; i++) 
			{
				line_mask[i] = 255;
				inv_line_mask[warp_col - 1 - i] = 255;
			}
			*/
			minMaxIdx(hist_peaks.colRange(0, midpoint), NULL, NULL, NULL, leftx_base_p); //histogram.colRange(0, midpoint)
			minMaxIdx(hist_peaks.colRange(midpoint, warp_col), NULL, NULL, NULL, rightx_base_p);
			__leftx_base = leftx_base_p[1];
			__rightx_base = rightx_base_p[1] + midpoint;
			// if (__first_sucs) // otherwise hist_width is not updated here
			hist_width = __rightx_base - __leftx_base;
		// }
		//else // disabled since only peaks are used for finding base, the restriction of hist_width may result in no peaks found
		//{
			//float cur_max = 0;
			//for (int dist = hist_width - 5; dist < hist_width + 5; dist ++)
			//{
				//for (int cur_left = midpoint - dist; cur_left < midpoint; cur_left ++)
				//{
					//if (cur_left < 0)
						//continue;
					//int cur_right = cur_left + dist;
					//if (cur_right > warp_col - 1)
						//break;
					//if ( histogram.at<float>(cur_left) + histogram.at<float>(cur_right) > cur_max)
					//{
						//__leftx_base = cur_left;
						//__rightx_base = cur_right;
						//cur_max = histogram.at<float>(cur_left) + histogram.at<float>(cur_right);
					//}
				//}
			//}
		//}
	}
	else if ( __split )
	{
		/// histogram to find the base for fitting
		Mat histogram(1, warp_col, CV_32FC1, Scalar(0));
		//float* line_his = histogram.ptr<float>();
		for (int i = 0; i < warp_row/4; i++ )
		{
			histogram += __warped_filter_image.row(i);
		}
		
		histogram= histogram.mul(__warped_filter_image.row(0));

				
		int midpoint = warp_col/2;
		double max_histogram_l, max_histogram_r;
		int left_center = __last_left_fit[2]*(warp_row - 1)*(warp_row - 1)+__last_left_fit[1]*(warp_row - 1) + __last_left_fit[0];
		int right_center = __last_right_fit[2]*(warp_row - 1)*(warp_row - 1)+__last_right_fit[1]*(warp_row - 1) + __last_right_fit[0];
		int left_l = max(0, int(left_center - __window_half_width));
		int left_r = min(warp_col, int(left_center + __window_half_width));
		int right_l = max(0, int(right_center - __window_half_width));
		int right_r = min(warp_col, int(right_center + __window_half_width));
		if ( left_r < 0 || right_l >= warp_col)
		{
			return;
		}
		
		minMaxIdx(histogram.colRange(left_l, left_r), NULL, &max_histogram_l, NULL, NULL);
		minMaxIdx(histogram.colRange(right_l, right_r), NULL, &max_histogram_r, NULL, NULL);
		
		cout << "max_histogram_l: " << max_histogram_l << "max_histogram_r: " << max_histogram_r << endl;
		cout << "left_r: " << left_r << ", right_l: " << right_l << endl;

		/// finding peaks and prefer peaks that do not have other peaks in between
		int min_peak_dist = warp_col / 20;
		float min_height_diff = (max_histogram_l + max_histogram_r)/4; // 5
		float min_height = min_height_diff*2;
		int mid_l = left_r, mid_r = right_l;
		double max_histogram_m;
		int max_histogram_m_idx;
		int max_new_m = 0;
		if (mid_l >= mid_r)
		{
			return;
		}
		vector<int> max_loc;
		vector<float> max_val;
		extractPeaks(histogram.colRange(mid_l, mid_r), min_peak_dist, min_height_diff, min_height, max_loc, max_val);
		// minMaxIdx(histogram.colRange(mid_l, mid_r), NULL, &max_histogram_m, NULL, &max_histogram_m_idx);
		float max_max_val_eva = 0, max_max_val = 0;
		int max_max_loc = 0;
		for (int i = 0; i < max_loc.size(); i++)
		{
			if (max_val[i]*sqrt(min(max_loc[i], mid_r - mid_l-1-max_loc[i])) > max_max_val_eva)
			{
				max_max_val = max_val[i];
				max_max_loc = max_loc[i];
			}
		}
		max_histogram_m = max_max_val;
		max_histogram_m_idx = max_max_loc;
		cout << "max_histogram_m: " << max_histogram_m << "max_histogram_m_idx: " << max_histogram_m_idx << endl;
		if (max_histogram_m > min_height )
		{
			max_new_m = max_histogram_m_idx + mid_l;
			__new_branch_found = true;
			__split = false;
			__split_recover_count = 0;
			__branch_grow_count = 22;

		}
		if (__new_branch_found)
		{
			if ( abs(left_center - __last_left_fit[0]) < abs(right_center -__last_right_fit[0]) )
			{
				__rightx_base = max_new_m;
				__leftx_base = 0;
				hist_width = __rightx_base - left_center;
				__branch_at_left = false;
			}
			else
			{
				__leftx_base = max_new_m;
				__rightx_base = 0;
				hist_width = right_center - __leftx_base;
				__branch_at_left = true;
			}
		}
		else
		{
			__leftx_base = 0;
			__rightx_base = 0;
		}
		
	}
	else if ( __branch_grow_count > 0)
	{
		if (__branch_at_left)
		{
			__leftx_base = __last_left_fit[2]*warp_row*warp_row+__last_left_fit[1]*warp_row + __last_left_fit[0];
			__leftx_base = (__leftx_base - warp_col/2)*(__k_pitch*warp_row + __b_pitch) + warp_col/2;
			__rightx_base = 0;
			float right_top = __last_right_fit[2]*warp_row *warp_row+__last_right_fit[1]*warp_row  + __last_right_fit[0];
			right_top = (right_top - warp_col/2)*(__k_pitch*warp_row + __b_pitch) + warp_col/2;
			hist_width = right_top - __leftx_base;
		}
		else
		{
			__rightx_base = __last_right_fit[2]*warp_row *warp_row+__last_right_fit[1]*warp_row  + __last_right_fit[0];
			__rightx_base = (__rightx_base - warp_col/2)*(__k_pitch*warp_row + __b_pitch) + warp_col/2;
			__leftx_base = 0;
			float left_top = __last_left_fit[2]*warp_row *warp_row+__last_left_fit[1]*warp_row  + __last_left_fit[0];
			left_top = (left_top - warp_col/2)*(__k_pitch*warp_row + __b_pitch) + warp_col/2;
			hist_width = __rightx_base - left_top;
		}
	}
	else
	{
		__leftx_base = 0;
		__rightx_base = 0; // then not used
	}
	#ifndef NDEBUG_FT
	cout << "leftx_base: " << __leftx_base << " , right_base: " << __rightx_base << endl;
	#endif
	return;
}


void LaneImage::__ROIInds(float half_width, valarray<float>& nonzx, valarray<float>& nonzy, valarray<bool>& left_lane_inds, valarray<bool>& right_lane_inds )
{
	float frow = (float)warp_row;
	float fcol = (float)warp_col;
	
	float window_height = warp_row/__window_number;
	if ( __initial_frame )	// originally only left
	{
		/// find pixels in windows from bottom up
		float leftx_cur = __leftx_base;
		float rightx_cur = __rightx_base;
		
		valarray<bool> good_left_inds, good_right_inds;

		float k_left = 0, k_right = 0, b_left = leftx_cur, b_right = rightx_cur;

		for (int i = 0; i < __window_number; i++)
		{
			float win_y_low = warp_row - (i+1)*window_height;
			float win_y_high = warp_row - i*window_height;
			float win_xleft_low = leftx_cur - half_width;
			float win_xleft_high = leftx_cur + half_width;
			float win_xright_low = rightx_cur - half_width;
			float win_xright_high = rightx_cur + half_width;
			good_left_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xleft_low) & (nonzx <= win_xleft_high);
			good_right_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xright_low) & (nonzx <= win_xright_high);
			
			left_lane_inds = left_lane_inds | good_left_inds;
			right_lane_inds = right_lane_inds | good_right_inds;
			
			valarray<float> nonzx_cur_left(nonzx[good_left_inds]);
			valarray<float> nonzx_cur_right(nonzx[good_right_inds]);
			// original method before 01/22/2018
			// if (nonzx_cur_left.size() > __window_min_pixel)
			// 	leftx_cur = nonzx_cur_left.sum()/nonzx_cur_left.size();
			// if (nonzx_cur_right.size() > __window_min_pixel)
			// 	rightx_cur = nonzx_cur_right.sum()/nonzx_cur_right.size();
			
			// new method considering scope
			valarray<float> nonzy_cur_left(nonzy[good_left_inds]);
			valarray<float> nonzy_cur_right(nonzy[good_right_inds]);

			findWindowPix(nonzx_cur_left, nonzy_cur_left, i, window_height, leftx_cur, __window_min_pixel);
			findWindowPix(nonzx_cur_right, nonzy_cur_right, i, window_height, rightx_cur, __window_min_pixel);
			
			
			
		}
	}
	else if ( __branch_grow_count > 0) 
	{
		if (__leftx_base != 0) // left is new
		{
			/// find pixels in windows from bottom up
			float leftx_cur = __leftx_base;
			valarray<bool> good_left_inds;
			for (int i = 0; i < __window_number - __branch_grow_count/3; i++)
			{
				float win_y_low = i*window_height;
				float win_y_high = (i+1)*window_height;
				float win_xleft_low = leftx_cur - half_width;
				float win_xleft_high = leftx_cur + half_width;
				good_left_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xleft_low) & (nonzx <= win_xleft_high);
				
				left_lane_inds = left_lane_inds | good_left_inds;
				
				valarray<float> nonzx_cur_left(nonzx[good_left_inds]);
				// // old method before 01/22/2018
				// if (nonzx_cur_left.size() > __window_min_pixel)
				// 	leftx_cur = nonzx_cur_left.sum()/nonzx_cur_left.size();

				// new method at 01/22/2018
				valarray<float> nonzy_cur_left(nonzy[good_left_inds]);
				findWindowPix(nonzx_cur_left, nonzy_cur_left, i, window_height, leftx_cur, __window_min_pixel);
			}
			valarray<float> right_ref = __last_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_right_fit[1]*(frow - 1 - nonzy) + __last_right_fit[0];
			right_ref = (right_ref - fcol/2)*(__k_pitch*(frow - 1 - nonzy) + __b_pitch) + fcol/2;
			right_lane_inds = (nonzx > (right_ref - half_width))
			& (nonzx < (right_ref + half_width)); // formula's y is from downside	
		}
		else if (__rightx_base != 0)
		{
			/// find pixels in windows from bottom up
			float rightx_cur = __rightx_base;
			valarray<bool> good_right_inds;
			for (int i = 0; i < __window_number - __branch_grow_count/3; i++)
			{
				float win_y_low = i*window_height;
				float win_y_high = (i+1)*window_height;
				float win_xright_low = rightx_cur - half_width;
				float win_xright_high = rightx_cur + half_width;
				good_right_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xright_low) & (nonzx <= win_xright_high);
				
				right_lane_inds = right_lane_inds | good_right_inds;
				
				valarray<float> nonzx_cur_right(nonzx[good_right_inds]);
				// // old method before 01/22/2018
				// if (nonzx_cur_right.size() > __window_min_pixel)
				// 	rightx_cur = nonzx_cur_right.sum()/nonzx_cur_right.size();
				// new method at 01/22/2018
				valarray<float> nonzy_cur_right(nonzy[good_right_inds]);
				findWindowPix(nonzx_cur_right, nonzy_cur_right, i, window_height, rightx_cur, __window_min_pixel);
			}
			valarray<float> left_ref = __last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0];
			left_ref = (left_ref - fcol/2)*(__k_pitch*(frow - 1 - nonzy) + __b_pitch) + fcol/2;
			left_lane_inds = (nonzx > (left_ref - half_width))
			& (nonzx < (left_ref + half_width)); // formula's y is from downside	
			// left_lane_inds = (nonzx > (__last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0] - half_width))
			// & (nonzx < (__last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0] + half_width));	
		}
		__branch_grow_count--;
	}
	else
	{
		valarray<float> left_ref = __last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0];
		left_ref = (left_ref - fcol/2)*(__k_pitch*(frow - 1 - nonzy) + __b_pitch) + fcol/2;
		left_lane_inds = (nonzx > (left_ref - half_width))
		& (nonzx < (left_ref + half_width)); // formula's y is from downside // consider pitch compensation

		valarray<float> right_ref = __last_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_right_fit[1]*(frow - 1 - nonzy) + __last_right_fit[0];
		right_ref = (right_ref - fcol/2)*(__k_pitch*(frow - 1 - nonzy) + __b_pitch) + fcol/2;
		right_lane_inds = (nonzx > (right_ref - half_width))
		& (nonzx < (right_ref + half_width)); // formula's y is from downside // consider pitch compensation

		// // originally not considering pitch compen, commented at 10/22/2017
		// left_lane_inds = (nonzx > (__last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0] - half_width))
		// & (nonzx < (__last_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_left_fit[1]*(frow - 1 - nonzy) + __last_left_fit[0] + half_width));
		// right_lane_inds = (nonzx > (__last_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_right_fit[1]*(frow - 1 - nonzy) + __last_right_fit[0] - half_width))
		// & (nonzx < (__last_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__last_right_fit[1]*(frow - 1 - nonzy) + __last_right_fit[0] + half_width)); // formula's y is from downside
		
		/*
		left_lane_inds = (nonzx > (__avg_hist_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__avg_hist_left_fit[1]*(frow - 1 - nonzy) + __avg_hist_left_fit[0] - half_width))
		& (nonzx < (__avg_hist_left_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__avg_hist_left_fit[1]*(frow - 1 - nonzy) + __avg_hist_left_fit[0] + half_width));
		right_lane_inds = (nonzx > (__avg_hist_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__avg_hist_right_fit[1]*(frow - 1 - nonzy) + __avg_hist_right_fit[0] - half_width))
		& (nonzx < (__avg_hist_right_fit[2]*((frow - 1 - nonzy)*(frow - 1 - nonzy))+__avg_hist_right_fit[1]*(frow - 1 - nonzy) + __avg_hist_right_fit[0] + half_width));
		*/
	}
	return;
}

void findWindowPix(valarray<float>& nonzx_cur_left, valarray<float>& nonzy_cur_left, int i, float window_height, float& leftx_cur, float __window_min_pixel)
{
	int num_left = nonzx_cur_left.size();
	if (num_left > __window_min_pixel/2 )
	{
		int sum_xi_left = nonzx_cur_left.sum();
		int sum_yi_left = nonzy_cur_left.sum();
		int sum_xi2_left = pow(nonzx_cur_left, (float)2).sum();
		int sum_yi2_left = pow(nonzy_cur_left, (float)2).sum();
		int sum_xiyi_left = (nonzx_cur_left * nonzy_cur_left).sum();

		Mat AtA_left(2, 2, CV_32FC1);
		AtA_left.at<float>(0, 0) = num_left;
		AtA_left.at<float>(0, 1) = sum_yi_left;
		AtA_left.at<float>(1, 0) = sum_yi_left;
		AtA_left.at<float>(1, 1) = sum_yi2_left;
		Mat AtA_left_inv = AtA_left.inv();

		float b_left = AtA_left_inv.at<float>(0, 0) * sum_xi_left + AtA_left_inv.at<float>(0, 1) * sum_xiyi_left;
		float k_left = AtA_left_inv.at<float>(1, 0) * sum_xi_left + AtA_left_inv.at<float>(1, 1) * sum_xiyi_left;

		leftx_cur = k_left * (warp_row - (i + 1.5) * window_height) + b_left;
	}
	return;
}

void LaneImage::__makeUpFilter(bool left, Mat& warped_filter_image_U, vector<Point>& nonz_loc, valarray<float>& nonzx, valarray<float>& nonzy, 
	int& hist_width, valarray<float>& leftx, valarray<float>& lefty, valarray<float>& rightx, valarray<float>& righty, VehMask& veh_masker, 
	int& grad_num_left, int& grad_num_right)
{
	// new method for makeup 0113
	int dilate_width = max( (int)round(__window_half_width*0.4),4);
	int move_dist = max( (int)round(__window_half_width*0.2),2);
	Mat expand_kernel = getStructuringElement(MORPH_RECT, Size(dilate_width, 1));

	{
		// Mat sobel_xp, sobel_xn;
		// threshold(sobelx, sobel_xp, 0, 1, THRESH_TOZERO);
		// sobel_xn = -sobelx;
		// threshold(sobel_xn, sobel_xn, 0, 1, THRESH_TOZERO);

		// sobelAbsThresh(sobel_xp, __filter_binary_x_p, __abs_x_thresh_pre);   // relative threshold is not good!!!
		// sobelAbsThresh(sobel_xn, __filter_binary_x_n, __abs_x_thresh_pre);
		__filter_binary_x_p = __sobelx > 300;
		__filter_binary_x_n = -__sobelx > 300;
		
		__filter_binary_x_p = __filter_binary_x_p & __filter_binary_dir;
		__filter_binary_x_n = __filter_binary_x_n & __filter_binary_dir;   // for __makeUpFilter

		cout << "filter_binary_p_2left size: " << __filter_binary_x_p.size() << endl;
		cout << "move_dist: " << move_dist << ", warp_col: " << warp_col << endl;

		Mat filter_binary_p_2left(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
		Mat filter_binary_n_2left(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
		Mat filter_binary_p_2right(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
		Mat filter_binary_n_2right(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0));
		
		filter_binary_p_2right.colRange(move_dist, warp_col) = __filter_binary_x_p.colRange(0, warp_col - move_dist) + 0;
		filter_binary_p_2left.colRange(0, warp_col - move_dist) = __filter_binary_x_p.colRange(move_dist, warp_col) + 0;
		filter_binary_n_2right.colRange(move_dist, warp_col) = __filter_binary_x_n.colRange(0, warp_col - move_dist) + 0;
		filter_binary_n_2left.colRange(0, warp_col - move_dist) = __filter_binary_x_n.colRange(move_dist, warp_col) + 0;
		

		dilate(filter_binary_p_2right, filter_binary_p_2right, expand_kernel);
		dilate(filter_binary_p_2left, filter_binary_p_2left, expand_kernel);
		dilate(filter_binary_n_2right, filter_binary_n_2right, expand_kernel);
		dilate(filter_binary_n_2left, filter_binary_n_2left, expand_kernel);

		__filter_binary_x_p.setTo(0, filter_binary_n_2left | filter_binary_n_2right);
		__filter_binary_x_n.setTo(0, filter_binary_p_2left | filter_binary_p_2right);
		
	}
	//////////////////////////////////////////////////////////////////////

	Mat filter_binary_half(__filter_binary_x_p.size(), __filter_binary_x_p.type(), Scalar(0)); // store only one-side-edge
	if (left)
	{
		int to_n_count = countNonZero(__filter_binary_x_n.colRange(0, warp_col/2));
		int to_p_count = countNonZero(__filter_binary_x_p.colRange(0, warp_col/2));
		if (to_n_count > to_p_count)
		{
			filter_binary_half.colRange(0, warp_col/2) = __filter_binary_x_n.colRange(0, warp_col/2) + 0; // left: bright-to-dark, sobelx is negative
			grad_num_left += to_n_count;
		}
		else
		{
			filter_binary_half.colRange(0, warp_col/2) = __filter_binary_x_p.colRange(0, warp_col/2) + 0; // left: bright-to-dark, sobelx is negative
			grad_num_left += to_p_count;
		}
	}	
	else
	{
		int to_n_count = countNonZero(__filter_binary_x_n.colRange(warp_col/2, warp_col));
		int to_p_count = countNonZero(__filter_binary_x_p.colRange(warp_col/2, warp_col));
		if (to_n_count > to_p_count)
		{
			filter_binary_half.colRange(warp_col/2, warp_col) = __filter_binary_x_n.colRange(warp_col/2, warp_col) + 0;
			grad_num_right += to_n_count;
		}
		else
		{
			filter_binary_half.colRange(warp_col/2, warp_col) = __filter_binary_x_p.colRange(warp_col/2, warp_col) + 0;
			grad_num_right += to_p_count;
		}
	}
	// new method for makeup 0113
	__warped_filter_image.setTo(0.5, filter_binary_half);
	///////////////////////////////
	
	// original method before 0113
	// __sobelx.setTo(0, ~filter_binary_half); // __sobelx is the original sobelx before this step
	
	// __sobelx.setTo(0, veh_masker.warp_veh_mask);

	// Mat abs_sobelx = abs(__sobelx);
	
	// Mat sobel_xtop(warp_row, warp_col, CV_32FC1, Scalar(0));
	// Mat sobel_xbot(warp_row, warp_col, CV_32FC1, Scalar(0));
	// sobel_xtop.rowRange(0, warp_row/2) = abs_sobelx.rowRange(0, warp_row/2) + 0;
	// sobel_xbot.rowRange(warp_row/2, warp_row) = abs_sobelx.rowRange(warp_row/2, warp_row) + 0;
	
	// Mat filter_binary_x_top, filter_binary_x_bot;
	// sobelAbsThresh(sobel_xtop, filter_binary_x_top, __abs_x_thresh);
	// sobelAbsThresh(sobel_xbot, filter_binary_x_bot, __abs_x_thresh);
	// __warped_filter_image.setTo(0.5, filter_binary_x_top | filter_binary_x_bot);

	// // __warped_filter_image.setTo(0, veh_masker.warp_veh_mask);
	
	#ifndef NDEBUG
	imshow("overall binary make-up", __warped_filter_image);
	//waitKey(0);
	#endif
	
	
	//Mat warped_filter_image_U;
	__warped_filter_image.convertTo(warped_filter_image_U, CV_8U, 255, 0 );
	int from_to[] = {0,0, 0,1, 0,2};
	Mat out[] = {__lane_window_out_img, __lane_window_out_img, __lane_window_out_img};
	mixChannels(&warped_filter_image_U, 1, out, 3, from_to, 3);
	
	float frow = (float)warp_row;
	
	/// find all non-zero pixels
	//vector<Point> nonz_loc;
	findNonZero(warped_filter_image_U, nonz_loc);
	
	cout << "# of non-zero pixels: " << nonz_loc.size() << endl;
	
	//valarray<float> nonzx(nonz_loc.size()), nonzy(nonz_loc.size());
	nonzx.resize(nonz_loc.size());
	nonzy.resize(nonz_loc.size());
	
	for (vector<Point>::iterator it = nonz_loc.begin() ; it != nonz_loc.end(); it++)
	{
		nonzx[it - nonz_loc.begin()] = (*it).x;
		nonzy[it - nonz_loc.begin()] = (*it).y;
	}
	valarray<bool> left_lane_inds(false, nonzx.size());
	valarray<bool> right_lane_inds(false, nonzx.size());
	
	/// find interested pixels
	__laneBase(hist_width);
	__ROIInds(__window_half_width, nonzx, nonzy, left_lane_inds, right_lane_inds);
	
	leftx = valarray<float>(nonzx[left_lane_inds]);
	lefty = valarray<float>(nonzy[left_lane_inds]);
	rightx = valarray<float>(nonzx[right_lane_inds]);
	righty = valarray<float>(nonzy[right_lane_inds]);
	
	//valarray<float> leftx(nonzx[left_lane_inds]);
	//valarray<float> lefty(nonzy[left_lane_inds]);
	//valarray<float> rightx(nonzx[right_lane_inds]);
	//valarray<float> righty(nonzy[right_lane_inds]);
	
	cout << "# of left pixels:  " << leftx.size() << endl;
	cout << "# of right pixels: " << rightx.size() << endl;
}






// void LaneImage::get_vanishing_point(Mat inv_per_mtx)		// now van_pt is not decided by lane detection result
// {
// 	float frow = (float)warp_row;
	
// 	valarray<bool> left_lane_inds = __lefty > frow/2;
// 	valarray<bool> right_lane_inds = __righty > frow/2;
	
// 	valarray<float> leftx(__leftx[left_lane_inds]);
// 	valarray<float> rightx(__rightx[right_lane_inds]);
// 	valarray<float> lefty(__lefty[left_lane_inds]);
// 	valarray<float> righty(__righty[right_lane_inds]);
	
// 	vector<Point2f> ori_pts_van;
	
// 	if (leftx.size() > __leftx.size()/4 && rightx.size() > __rightx.size()/4) // pixels in lower part are more than a threshold
// 	{
// 		size_t length_left = leftx.size();
// 		size_t length_right = rightx.size();
			
// 		Mat X_left(1, length_left, CV_32F);
// 		Mat X_right(1, length_right, CV_32F);
// 		Mat Y_left(2, length_left, CV_32F, Scalar_<float>(1));
// 		Mat Y_right(2, length_right, CV_32F, Scalar_<float>(1));
			
// 		float* line_Yl1 = Y_left.ptr<float>(1);
// 		float* line_Xl = X_left.ptr<float>();
// 		float* line_Yr1 = Y_right.ptr<float>(1);
// 		float* line_Xr = X_right.ptr<float>();
			
// 		for (int i = 0; i < length_left; i++)
// 		{
// 			line_Yl1[i] = (frow - 1 - lefty[i]); // from downside
// 			line_Xl[i] = leftx[i];
// 		}
// 		for (int i = 0; i < length_right; i++)
// 		{
// 			line_Yr1[i] = (frow - 1 -righty[i]);
// 			line_Xr[i] = rightx[i];
// 		}
// 		// fit linear model to the down-half of selected pixels for finding vanishing point for next frame
				
// 		Mat left_fit = (Y_left.t()).inv(DECOMP_SVD)*(X_left.t());
// 		__left_fit_2 = left_fit;
// 		Mat right_fit = (Y_right.t()).inv(DECOMP_SVD)*(X_right.t());
// 		__right_fit_2 = right_fit;
		
// 		ori_pts_van.push_back(Point2f(__left_fit_2[1]*(frow - 1 - lefty.max()) + __left_fit_2[0], lefty.max())); // the y in the linear model is different from the real y index
// 		ori_pts_van.push_back(Point2f(__left_fit_2[1]*(frow - 1 - lefty.min()) + __left_fit_2[0], lefty.min()));
// 		ori_pts_van.push_back(Point2f(__right_fit_2[1]*(frow - 1 - righty.max()) + __right_fit_2[0], righty.max()));
// 		ori_pts_van.push_back(Point2f(__right_fit_2[1]*(frow - 1 - righty.min()) + __right_fit_2[0], righty.min()));
// 	}
// 	else
// 	{
// 		ori_pts_van.push_back(Point2f(__left_fit_2[1]*(frow - 1 - __lefty.max()) + __left_fit_2[0], __lefty.max())); // the y in the linear model is different from the real y index
// 		ori_pts_van.push_back(Point2f(__left_fit_2[1]*(frow - 1 - __lefty.min()) + __left_fit_2[0], __lefty.min()));
// 		ori_pts_van.push_back(Point2f(__right_fit_2[1]*(frow - 1 - __righty.max()) + __right_fit_2[0], __righty.max()));
// 		ori_pts_van.push_back(Point2f(__right_fit_2[1]*(frow - 1 - __righty.min()) + __right_fit_2[0], __righty.min()));
// 	}
	
	
// 	vector<Point2f> trs_pts_van;
// 	perspectiveTransform(ori_pts_van, trs_pts_van, inv_per_mtx);
	
// 	__left_fit_2_img[1] = - (trs_pts_van[1].x - trs_pts_van[0].x)/(trs_pts_van[1].y - trs_pts_van[0].y); // The k in x = ky+b.negative because of the flipped y
// 	__left_fit_2_img[0] = trs_pts_van[1].x - __left_fit_2_img[1]*(frow - 1 - trs_pts_van[1].y);
// 	__right_fit_2_img[1] = - (trs_pts_van[3].x - trs_pts_van[2].x)/(trs_pts_van[3].y - trs_pts_van[2].y); // The k in x = ky+b.negative because of the flipped y
// 	__right_fit_2_img[0] = trs_pts_van[3].x - __right_fit_2_img[1]*(frow - 1 - trs_pts_van[3].y);
	
// 	/*
// 	cout << "left linear model:  " << __left_fit_2_img << endl;
// 	cout << "right linear model: " << __right_fit_2_img << endl;
// 	*/
	
// 	__van_pt.y = - (__left_fit_2_img[0] - __right_fit_2_img[0])/(__left_fit_2_img[1] - __right_fit_2_img[1]);
// 	__van_pt.x = __left_fit_2_img[1]*__van_pt.y + __left_fit_2_img[0];
// 	__van_pt.y = frow - 1 - __van_pt.y;
	
// 	cout << "vanishing point: " << __van_pt << endl;
	
// 	// if points are few, using linear model to replace second order model
// 	if (__lefty.min() > frow/2)
// 	{
// 		__left_fit[2] = 0;
// 		__left_fit[1] = __left_fit_2[1];
// 		__left_fit[0] = __left_fit_2[0];
// 		__left_fit_cr[2] = 0;
// 		__left_fit_cr[1] = __left_fit[1]*xm_per_pix/ym_per_pix;
// 		__left_fit_cr[0] = __left_fit[0]*xm_per_pix/ym_per_pix/ym_per_pix;
// 		cout << "Left lane using linear model. " << endl;
// 	}
// 	if (__righty.min() > frow/2)
// 	{
// 		__right_fit[2] = 0;
// 		__right_fit[1] = __right_fit_2[1];
// 		__right_fit[0] = __right_fit_2[0];
// 		__right_fit_cr[2] = 0;
// 		__right_fit_cr[1] = __right_fit[1]*xm_per_pix/ym_per_pix;
// 		__right_fit_cr[0] = __right_fit[0]*xm_per_pix/ym_per_pix/ym_per_pix;
// 		cout << "Right lane using linear model. " << endl;
// 	}
	
// 	return;
	
// }


void LaneImage::__laneSanityCheck(int hist_width) // consistent with function getLaneWidthWarp
{

	// /// average lane width in warped image			// parallel check disabled because of the model is embedded with parallelism
	// float y_eval_loc = 0; // changed to counting from closer side
	// valarray<float> y_eval(y_eval_loc, 10);
	// float step = warp_row/10;
	// for (int i = 1; i<10; i++)
	// 	y_eval[i] = y_eval[i-1] + step; // changed to counting from closer side
	// valarray<float> x_l(10);
	// valarray<float> x_r(10);
	// x_l = __left_fit[2]*y_eval*y_eval + __left_fit[1]*y_eval + __left_fit[0];
	// x_r = __right_fit[2]*y_eval*y_eval + __right_fit[1]*y_eval + __right_fit[0];
	// valarray<float> x_dist = abs(x_r-x_l);
	// __dif_dist = x_dist.max()/x_dist.min();
	
	// valarray<float> curvature_l(10);
	// valarray<float> curvature_r(10);
	// curvature_l = abs(2*__left_fit[2])/pow(( 1 + pow(( 2*__left_fit[2]*y_eval + __left_fit[1]), 2)),1.5);
	// curvature_r = abs(2*__right_fit[2])/pow(( 1 + pow(( 2*__right_fit[2]*y_eval + __right_fit[1]), 2)),1.5);
	// valarray<float> curve_dist = abs(curvature_l-curvature_r);
	// valarray<float> curve_time_a = curvature_l/curvature_r;
	// valarray<float> curve_time_b = curvature_r/curvature_l;
	// __dif_curve = curve_dist.max();
	// __time_curve = max( curve_time_a.max(), curve_time_b.max() );
	
	// cout << "Left curvature: ";
	// for (int i = 0; i < 10; i++)
	// {
	// 	cout << curvature_l[i] << " ";
	// }
	// cout << endl;
	// cout << "Right curvature: ";
	// for (int i = 0; i < 10; i++)
	// {
	// 	cout << curvature_r[i] << " ";
	// }
	// cout << endl;
	
	// cout << "Change of lane width: " << __dif_dist << "Diff of curvature: " << __dif_curve << "Time of curve: " << __time_curve << endl;

	// __parallel_check = ( (abs(__left_fit[2]-__right_fit[2])+abs(__left_fit[1]-__right_fit[1])) < 0.2) && __dif_dist < 1.3 && (__dif_curve < 0.0004 || __time_curve < 2); // can change later
		__parallel_check = true;
		/*
		float dist2left = left_lane.line_base_pos.back();
		float dist2right = right_lane.line_base_pos.back();
        bool lane_width_check = ((abs(dist2right) + abs(dist2left)) < 3.7) && ((abs(dist2right) + abs(dist2left)) > 3.0);
        */
        __bot_width = abs(__left_fit[0]-__right_fit[0]);
        __width_check = ( __bot_width > 0.8 * hist_width && __bot_width < 1.2*hist_width  && __bot_width > __min_width_warp); // __bot_width > warp_col/8
		#ifndef NDEBUG_FT
		cout << "para check" << __parallel_check << "width check" << __width_check << endl;
		cout << "__bot_width: " << __bot_width << ", __min_width_warp: " << __min_width_warp << ", hist_width: " << hist_width << endl;
		#endif
        // width check disabled temporally
		return ;// && lane_width_check;

}

float LaneImage::__getCurveDiff(Vec3f& cur_fit, Vec3f& hist_fit) // consistent with function getLaneWidthWarp
{
	/// average lane width in warped image
	float y_eval_loc = warp_row/2; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 6);
	float step = warp_row/10;
	for (int i = 1; i<6; i++)
		y_eval[i] = y_eval[i-1] + step; // changed to counting from closer side
	valarray<float> curvature_l(6);
	valarray<float> curvature_r(6);
	curvature_l = (2*cur_fit[2])/pow(( 1 + pow(( 2*cur_fit[2]*y_eval + cur_fit[1]), 2)),1.5);
	curvature_r = (2*hist_fit[2])/pow(( 1 + pow(( 2*hist_fit[2]*y_eval + hist_fit[1]), 2)),1.5);  // save the sign
	
	valarray<float> x_dist = abs(curvature_l-curvature_r);
	float dif_max = x_dist.max();
	
	return dif_max;

}

float LaneImage::__getDiff(Vec3f& cur_fit, Vec3f& hist_fit) // max distance of current result from history
{
	/*
	/// average lane width in warped image
	float y_eval_loc = 0; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 11);
	float step = warp_row/10;
	for (int i = 1; i<11; i++)
		y_eval[i] = y_eval[i-1] + step; // changed to counting from closer side
	valarray<float> x_l(11);
	valarray<float> x_r(11);
	x_l = cur_fit[2]*y_eval*y_eval + cur_fit[1]*y_eval + cur_fit[0];
	x_r = hist_fit[2]*y_eval*y_eval + hist_fit[1]*y_eval + hist_fit[0];
	valarray<float> x_dist = abs(x_r-x_l);
	float dif_max = x_dist.max();
	*/
	float dif_max = abs(cur_fit[2] - hist_fit[2]);
	if (cur_fit[2] * hist_fit[2] < 0)
		dif_max = - dif_max;
	
	return dif_max;// && lane_width_check;

}


void extractPeaks(const Mat& src, const int min_peak_dist, const float min_height_diff, const float min_height, vector<int>& max_loc, vector<float>& max_val)
{
	int ker_size = min_peak_dist;
	Point cur_max_pt;
	Point cur_min_pt;
	double cur_max;
	double cur_min;
	
	max_loc.clear();
	max_val.clear();
	/// find local maxima according to rule
	for (int loc = ker_size; loc < src.cols - ker_size; loc++)
	{
		Mat part = src.colRange(loc-ker_size, loc+ker_size);
		minMaxLoc(part, &cur_min, &cur_max, NULL, &cur_max_pt);
		if (cur_max_pt.x == ker_size && cur_max - cur_min >= min_height_diff && cur_max >= min_height )
		{
			max_loc.push_back(loc);
			max_val.push_back(cur_max);
		}
	}
	
	int num_max = max_loc.size();
	if (num_max <= 0)
	{
		cout << "No peaks found. Use maximal value instead: ";
		minMaxLoc(src, &cur_min, &cur_max, NULL, &cur_max_pt);
		max_loc.push_back(cur_max_pt.x);
		max_val.push_back(cur_max);
		cout << "[" << max_loc[0] << ", " << max_val[0] << "] " << endl;
	}
	else
	{
		cout << num_max << "Peaks found. ";
		for (int i = 0; i < num_max; i++)
		{
			cout << "[" << max_loc[i] << ", " << max_val[i] << "] ";
		}
		cout << endl;
	}
}
