#include "LaneImage.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

//float ym_per_pix = 40./720.;
//float xm_per_pix = 3.7/600.;
float ym_per_pix = 40./500.;
float xm_per_pix = 3.7/200.;

int warp_col = 400;
int warp_row = 500;

#ifdef DTREE
LaneImage::LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp, 
	Vec3f left_fit, Vec3f right_fit, Vec3f avg_hist_left_fit, Vec3f avg_hist_right_fit, vector<int> chnl_thresh, Ptr<ml::DTrees> BGR_tree, Ptr<ml::DTrees> HLS_tree, Mat dist_coeff, Mat cam_mtx )
#endif
#ifdef LREGG
LaneImage::LaneImage (Mat& per_mtx, Mat& inv_per_mtx, Mat& image, float nframe, int samp_cyc, int ini_flag, int& hist_width, bool first_sucs, int window_half_width, Mat& BGR_sample, Mat& HLS_sample, Mat& BGR_resp, Mat& HLS_resp, 
	Vec3f left_fit, Vec3f right_fit, Vec3f avg_hist_left_fit, Vec3f avg_hist_right_fit, vector<int> chnl_thresh, Ptr<ml::LogisticRegression> BGR_regg, Ptr<ml::LogisticRegression> HLS_regg, Mat dist_coeff, Mat cam_mtx )
#endif
{
	__raw_image = image;
	__row = image.rows;
	__col = image.cols;
	__calibration_dist = dist_coeff;
	__calibration_mtx = cam_mtx;
	__calibration();
	
	//__sobel_kernel_size = 15;
	__sobel_kernel_size = max(warp_col/80, 5 );
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
		
		cout << "Color thresh: " << __b_thresh[0] << " " << __g_thresh[0] << " " << __r_thresh[0] << " " << __h_thresh[1] << " " << __l_thresh[0] << " " << __s_thresh[0] << " " << endl;
		cout << "Adaptive color threshold used. " << endl;
	}
	
	#ifdef DTREE
	if ( BGR_tree.empty() )
	{
		__BGR_tree = ml::DTrees::create();
		cout << "new BGR tree." << endl;
	}
	else
	{
		__BGR_tree = BGR_tree;
	}
	if ( HLS_tree.empty() )
	{
		__HLS_tree = ml::DTrees::create();
		cout << "new HLS tree." << endl;
	}
	else
	{
		__HLS_tree = HLS_tree;
		
	}
	#endif
	#ifdef LREGG
	if ( BGR_regg.empty() )
	{
		__BGR_regg = ml::LogisticRegression::create();
		cout << "new BGR regg." << endl;
	}
	else
	{
		__BGR_regg = BGR_regg;
	}
	if ( HLS_regg.empty() )
	{
		__HLS_regg = ml::LogisticRegression::create();
		cout << "new HLS regg." << endl;
	}
	else
	{
		__HLS_regg = HLS_regg;
		
	}
	#endif
	
	__BGR_sample = BGR_sample;
	__HLS_sample = HLS_sample;
	__BGR_resp = BGR_resp;
	__HLS_resp = HLS_resp;
	__samp_cyc = samp_cyc;
	__nframe = nframe;
	
	__left_nolane = false;
	__right_nolane = false;
	
	__window_number = 10;
	//__window_half_width = warp_col/12;
	__window_half_width = window_half_width;
	__window_min_pixel = warp_row/__window_number*__window_half_width*2/100; // 1%.   60 for 1280*720(hand set). 
	cout << "min pixel: " << __window_min_pixel << endl;
	
	__last_left_fit = left_fit;
	__last_right_fit = right_fit;
	__left_fit = Vec3f(0, 0, 0);
	__right_fit = Vec3f(0, 0, 0);
	__left_fit_cr = Vec3f(0, 0, 0);
	__right_fit_cr = Vec3f(0, 0, 0);
	
	__left_dist_to_hist = 0;
	__right_dist_to_hist = 0;
	__left_curve_dist_to_hist = 0;
	__right_curve_dist_to_hist = 0;
	__avg_hist_left_fit = avg_hist_left_fit;
	__avg_hist_right_fit = avg_hist_right_fit;
	
	__first_sucs = first_sucs;
	
	if (ini_flag == 0)
	{
	clock_t t_last = clock();
	__transform_matrix = per_mtx;
	__warp();
	
	clock_t t_now = clock();
	cout << "Image warpped, using " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
	t_last = t_now;
	
	//__filter_binary = Mat::ones(__row, __col, CV_32F);
	__warped_filter_image = Mat::ones(warp_row, warp_col, CV_32F);
	__imageFilter();
	
	t_now = clock();
	cout << "Image filtered, using " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
	t_last = t_now;
	
	__lane_window_out_img = Mat(warp_row, warp_col, CV_8UC3, Scalar(0, 0, 0));
	__fitLaneMovingWindow(hist_width);
	
	t_now = clock();
	cout << "Image fitted, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
	t_last = t_now;
	
	if (__left_fit != Vec3f(0, 0, 0) && __right_fit != Vec3f(0, 0, 0))
	{
		//__getLaneWidthWarp();
		get_vanishing_point(inv_per_mtx);
	}
	
	t_now = clock();
	cout << "New vanishing point found, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
	t_last = t_now;
	}
	else
		cout << "Current frame is not processed due to failed initialization. " << endl;
}




int initialVan(Mat& raw_img, Mat& image, Mat& edges, Point2f& ini_van_pt, const vector<Point>& series_van_pt, const Point& avg_series_van, const Point2f& last_van_pt, int& y_bottom_warp, vector<int>& local_min_pt, float& illu_comp, vector<vector<Point> >& warp_test_vec, bool sucs_before, float track_radius)
{	
	//getchar();
	/*
	int a = image.at<uchar>(400, 400);
	cout << "current value: " << a << endl;
	Mat image_aug;
	image_aug = image*50;
	a = image_aug.at<uchar>(400, 400);
	cout << "current value: " << a << endl;
	getchar();
	*/
	
	/// Canny and edge filter
	Canny(image, edges, 50, 150, 3);
	
	#ifndef NDEBUG_IN
	imshow("canny", edges);
	#endif
	
	Mat sobel_x, sobel_y, sobel_angle;
	Sobel(image, sobel_x, CV_32F, 1, 0, 3);
	Sobel(image, sobel_y, CV_32F, 0, 1, 3);
	phase(sobel_x, sobel_y, sobel_angle, true);  // output of phase in degree is [0~360]
	Mat angle_mask;
	angle_mask = (sobel_angle >= 10 & sobel_angle <= 80) | (sobel_angle >= 100 & sobel_angle <= 170) | (sobel_angle >= 190 & sobel_angle <= 260) | (sobel_angle >= 280 & sobel_angle <= 350);
	//angle_mask = (sobel_angle >= 120 & sobel_angle <= 240);
	//#ifndef NDEBUG_IN
	//imshow("angle_mask", angle_mask);
	//waitKey(0);
	//#endif
	
	bitwise_and(edges, angle_mask, edges); // remove edges with wrong angle
	
	Mat cali_mask;
	Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(7, 7) );
	erode(image, cali_mask, erode_kernel );
	cali_mask = cali_mask > 0;
	bitwise_and(edges, cali_mask, edges); // remove the edges caused by warp effect
	
	#ifndef HIGH_BOT
	edges.rowRange(0, image.rows/2) = 0;
	#else
	edges.rowRange(0, image.rows/2) = 0;
	edges.rowRange(image.rows*7/10, image.rows) = 0;  // for caltech data
	#endif
	
	if (sucs_before)
	{
		//Mat mask_trap_previous(image.size(), CV_8UC1, Scalar(0));
		//drawContours(mask_trap_previous, warp_test_vec, -1, Scalar(255), CV_FILLED );
		//edges = edges & mask_trap_previous; // remove edges out of the trapezodial area
		edges.rowRange(0, warp_test_vec[0][0].y) = 0;
		edges.rowRange(warp_test_vec[0][1].y, image.rows) = 0;
		
	}
	
	
	
	
	
	
	
	
	/// vote for vanishing point based on Hough
	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI/180, 10, 10, 10 );
	//HoughLinesP(edges, lines, 20, CV_PI/180*3, 30, 10, 50 );
	
	if (lines.size() <= 0) /// safe return
	{/*
		van_pt = hist_van_pt; // y_bottom_warp, local_min_pt are taken care of by historical (or default) value. 
		float y_top_warp = (y_bottom_warp + 5*van_pt.y)/6;
		float x_van = van_pt.x;
		float y_van = van_pt.y;
		float y_bottom = y_bottom_warp;
		float x_left = 0;
		float x_right = image.cols - 1;
		//vector<vector<Point> > warp_test_vec;
		vector<Point> warp_test;
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_left) + x_left, y_top_warp));
		warp_test.push_back(Point(x_left, y_bottom ));
		warp_test.push_back(Point(x_right, y_bottom ));
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_right)+ x_right, y_top_warp));
		warp_test_vec.push_back(warp_test);
		*/
		cout << "Initilization failed: no Hough lines found. " << endl;
		return -1;
	}
	
	Mat vote_left(image.rows, image.cols, CV_64FC1, Scalar(0));
	Mat vote_right(image.rows, image.cols, CV_64FC1, Scalar(0));
	Mat vote_line(image.rows, image.cols, CV_8UC1, Scalar(0));  // contain lines qualified to vote, decide bottom and sample color threshold 
	
	float y_bottom_left = 0, y_bottom_right= 0; // warp source bottom
	
	#ifndef HIGH_BOT
	float y_bottom_max = min(image.rows * 14/15, image.rows -1 );
	#else
	float y_bottom_max = min(image.rows *7/10, image.rows -1 );  // for caltech data
	#endif
	
	float van_avg_x = avg_series_van.x, van_avg_y = avg_series_van.y;
	
	#ifndef NDEBUG_IN
	Mat edges_w_hist_van;
	edges.copyTo(edges_w_hist_van);
	if (series_van_pt.size() == 5) 
		//circle(edges_w_hist_van, avg_series_van, track_radius*2, Scalar(255), 1);
		rectangle(edges_w_hist_van, Point(image.cols/4, avg_series_van.y - track_radius*2), avg_series_van + Point(image.cols*3/4, avg_series_van.y + track_radius*2), Scalar(255), -1);
	imshow("masked canny", edges_w_hist_van);
	#endif
		
	for (int i = 0; i < lines.size(); i++)
	{
		float x1 = lines[i][0];
		float y1 = lines[i][1];
		float x2 = lines[i][2];
		float y2 = lines[i][3];
		double w = (abs(x1-x2) +abs(y1-y2));
		float k = (x1-x2)/(y1-y2);
		
		//if (series_van_pt.size() == 5) // check if going through the historical vanishing point
		//{
			//float a = y2 - y1;
			//float b = x1 - x2;
			//float c = x2*y1 - x1*y2;
			//float dist = abs(a*van_avg_x + b*van_avg_y + c)/ sqrt(a*a + b*b);
			//if (dist > track_radius*2) // 50 if track radius is 20
				//continue;
		//}
		
		if (abs(k) < 0.3 || abs(k) > 3 )
			continue;
		float x0, y0; // find lower point
		if (y1 > y2)
		{
			x0 = x1;
			y0 = y1;
		}
		else
		{
			x0 = x2;
			y0 = y2;
		}
		if (k > 0) 	// right
		{
			if (x0 < image.cols / 2)
				continue;
			// for (int j = 0; j < y0 ; j++ )
			// {
			// 	int x_cur = x0 - k*j;
			// 	int y_cur = y0 - j;
			// 	if (x_cur > image.cols - 1 || x_cur < 0)
			// 		break;
			// 	vote_right.at<double>(y_cur, x_cur)+= w;
			// }
			if (x0 + k*(y_bottom_max - y0)> image.cols - 1) // not approaching bottom
			{
				float lower_y = y0 + (image.cols - 1 - x0)/k;
				if (lower_y > y_bottom_right)
					y_bottom_right = lower_y;
			}
			else
				y_bottom_right = y_bottom_max;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
		}
		else // left
		{
			if (x0 > image.cols / 2)
				continue;
			// for (int j = 0; j < y0 ; j++ )
			// {
			// 	int x_cur = x0 - k*j;
			// 	int y_cur = y0 - j;
			// 	if (x_cur > image.cols - 1 || x_cur < 0)
			// 		break;
			// 	vote_left.at<double>(y_cur, x_cur)+= w;
			// }
			if (x0 + k*(y_bottom_max - y0)< 0) // not approaching bottom
			{
				float lower_y = y0 + (0 - x0)/k;
				if (lower_y > y_bottom_left)
					y_bottom_left = lower_y;
			}
			else
				y_bottom_left = y_bottom_max;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
		}
	}
	/// calculate proper warp region (lower vertex)
	if ( !sucs_before ) // only find the bottom warp at first 
	{
		y_bottom_warp = y_bottom_max ;
		if ((y_bottom_left != y_bottom_max || y_bottom_right != y_bottom_max ) && y_bottom_left != 0 && y_bottom_right != 0)
		{
			y_bottom_warp = min(y_bottom_left, y_bottom_right);
		}
	}
	cout << "y_bottom_warp: " << y_bottom_warp << endl;
	
	
	Mat votemap = vote_left.mul(vote_right)/(vote_left + vote_right);
	
	#ifndef NDEBUG_IN
	imshow("vote left", vote_left);
	imshow("vote right", vote_right);
	//imshow("vote map", votemap);
	waitKey(0);
	#endif
	
	if (series_van_pt.size() == 5)
	{
		Mat mask_track(image.rows, image.cols, CV_8UC1, Scalar(255));
		//circle(mask_track, avg_series_van, track_radius*2, Scalar(0), -1);
		//rectangle(mask_track, Point(image.cols/4, avg_series_van.y - track_radius*2), Point(image.cols*3/4, avg_series_van.y + track_radius*2), Scalar(0), -1);
		rectangle(mask_track, Point(avg_series_van.x - track_radius*4, avg_series_van.y - track_radius*2), Point(avg_series_van.x + track_radius*4, avg_series_van.y + track_radius*2), Scalar(0), -1);
		votemap.setTo(0, mask_track); // remove the possible vanishing point out of the track radius*2
	}
	
	double maxval;
	Point van_pt_int;
	minMaxLoc(votemap, NULL, &maxval, NULL, &van_pt_int);
	ini_van_pt.x = (int)van_pt_int.x;
	ini_van_pt.y = (int)van_pt_int.y;
	cout << "first van pt: " << van_pt_int << "maxval: "<< maxval << endl;
	
	//if (sucs_before)
	//{
	//	float w_current_ini_y = 0.2, w_current_ini_x = 0.8;
	//	ini_van_pt.x = w_current_ini_x*(float)(ini_van_pt.x) + (1-w_current_ini_x)*last_van_pt.x;
	//	ini_van_pt.y = w_current_ini_y*(float)(ini_van_pt.y) + (1-w_current_ini_y)*last_van_pt.y; // avg with best known vanshing point or avg series van
	//}
	
	if (maxval == 0) /// safe return
	{/*
		van_pt = hist_van_pt; // y_bottom_warp, local_min_pt are taken care of by historical (or default) value. 
		float y_top_warp = (y_bottom_warp + 5*van_pt.y)/6;
		float x_van = van_pt.x;
		float y_van = van_pt.y;
		float y_bottom = y_bottom_warp;
		float x_left = 0;
		float x_right = image.cols - 1;
		//vector<vector<Point> > warp_test_vec;
		vector<Point> warp_test;
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_left) + x_left, y_top_warp));
		warp_test.push_back(Point(x_left, y_bottom ));
		warp_test.push_back(Point(x_right, y_bottom ));
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_right)+ x_right, y_top_warp));
		warp_test_vec.push_back(warp_test);
		*/
		cout << "Initilization failed: no vanishing point found. " << endl;
		return -1;
	}
	
	#ifndef NDEBUG_IN
	/// draw the vote mape
	Mat votemap_see;
	votemap.convertTo(votemap_see, CV_8U, 255.0/maxval, 0);
	circle(votemap_see, van_pt_int, 5, Scalar(255), -1);
	imshow("votemap", votemap_see);
	//circle(vote_line, van_pt_int, 5, Scalar(255), -1);
	imshow("vote_line", vote_line);
	waitKey(0);
	#endif
	
	/// generate the trapezoid for warping and masking
	float y_top_warp = (y_bottom_warp + 5*ini_van_pt.y)/6;
	float x_van = ini_van_pt.x;
	float y_van = ini_van_pt.y;
	float y_bottom = y_bottom_warp;
	float x_left = 0;
	float x_right = image.cols - 1;
	//vector<vector<Point> > warp_test_vec;
	vector<Point> warp_test;
	warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_left) + x_left, y_top_warp));
	warp_test.push_back(Point(x_left, y_bottom ));
	warp_test.push_back(Point(x_right, y_bottom ));
	warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_right)+ x_right, y_top_warp));
	warp_test_vec.clear();
	warp_test_vec.push_back(warp_test);
	Mat mask_trap(image.size(), CV_8UC1, Scalar(0));
	drawContours(mask_trap, warp_test_vec, -1, Scalar(255), CV_FILLED );
	
	
	/// generate thresholds for color channels
	Mat pixhis_b, pixhis_g, pixhis_r;
	Mat pixhis_h, pixhis_l, pixhis_s;
	Mat img_hls(raw_img.size(), CV_8UC3);
	cvtColor(raw_img, img_hls, COLOR_BGR2HLS);
	vector<Mat> bgr_chnl, hls_chnl;
	split(raw_img, bgr_chnl);
	split(img_hls, hls_chnl);
	
	Mat blur_edges;
	//GaussianBlur(edges, blur_edges, Size(9,9), 5, 5);
	Mat dilate_kernel = getStructuringElement(MORPH_RECT, Size(3,3) ); // 6*6 for big image, 3*3 for small image
	//dilate(edges, blur_edges, dilate_kernel); 
	dilate(vote_line, blur_edges, dilate_kernel); // reject the edges that are not qualified
	
	#ifndef HIGH_BOT
	blur_edges.rowRange(0, image.rows*2/3) = 0;
	#else
	blur_edges.rowRange(0, image.rows/2) = 0;
	blur_edges.rowRange(image.rows*7/10, image.rows) = 0;  // for caltech data
	#endif
	
	blur_edges = blur_edges & mask_trap;
	int num_nonz = countNonZero(blur_edges);
	
	#ifndef NDEBUG_IN
	imshow("blured canny", blur_edges );
	waitKey(0);
	#endif
	
	/// histogram 
	int hist_size = 256;
	float range_bgrls[] = {0, 256};
	const float* range_5 = {range_bgrls};
	int hist_size_h = 180;
	float range_h[] = {0, 180};
	const float* range_1 = {range_h};
	bool uniform = true;
	bool accumulate = false;
	
	calcHist( &bgr_chnl[0], 1, 0, blur_edges, pixhis_b, 1, &hist_size, &range_5, uniform, accumulate ); // or edges, height_mask, Mat()
	calcHist( &bgr_chnl[1], 1, 0, blur_edges, pixhis_g, 1, &hist_size, &range_5, uniform, accumulate );
	calcHist( &bgr_chnl[2], 1, 0, blur_edges, pixhis_r, 1, &hist_size, &range_5, uniform, accumulate );
	calcHist( &hls_chnl[0], 1, 0, blur_edges, pixhis_h, 1, &hist_size_h, &range_1, uniform, accumulate );
	calcHist( &hls_chnl[1], 1, 0, blur_edges, pixhis_l, 1, &hist_size, &range_5, uniform, accumulate );
	calcHist( &hls_chnl[2], 1, 0, blur_edges, pixhis_s, 1, &hist_size, &range_5, uniform, accumulate );
	
	GaussianBlur(pixhis_b, pixhis_b, Size(25,25), 10); // smooth the histograms
	GaussianBlur(pixhis_g, pixhis_g, Size(25,25), 10);
	GaussianBlur(pixhis_r, pixhis_r, Size(25,25), 10);
	GaussianBlur(pixhis_h, pixhis_h, Size(19,19), 7);
	GaussianBlur(pixhis_l, pixhis_l, Size(25,15), 10);
	GaussianBlur(pixhis_s, pixhis_s, Size(25,25), 10);
	
	// find local maximum and desired threshold (minimum between maximum)
	/*vector<vector<int> > local_max_pt(6);
	vector<vector<float> > local_max(6);
	localMaxima(pixhis_b, 20, 15, 0, local_max_pt[0], local_max[0], local_min_pt[0], "b");
	localMaxima(pixhis_g, 20, 15, 0, local_max_pt[1], local_max[1], local_min_pt[1], "g");
	localMaxima(pixhis_r, 20, 15, 0, local_max_pt[2], local_max[2], local_min_pt[2], "r");
	localMaxima(pixhis_h, 10, 15, 0, local_max_pt[3], local_max[3], local_min_pt[3], "h");
	localMaxima(pixhis_l, 25, 15, 0, local_max_pt[4], local_max[4], local_min_pt[4], "l");
	localMaxima(pixhis_s, 25, 15, 0, local_max_pt[5], local_max[5], local_min_pt[5], "s");
	*/
	localMaxima(pixhis_b, num_nonz, local_min_pt[0], hist_size);
	localMaxima(pixhis_g, num_nonz, local_min_pt[1], hist_size);
	localMaxima(pixhis_r, num_nonz, local_min_pt[2], hist_size);
	localMaxima(pixhis_h, num_nonz, local_min_pt[3], hist_size_h);
	localMaxima(pixhis_l, num_nonz, local_min_pt[4], hist_size);
	localMaxima(pixhis_s, num_nonz, local_min_pt[5], hist_size);
	
	cout << "finish finding peaks: " << local_min_pt[0] << " " << local_min_pt[1] << " " << local_min_pt[2] << " " << local_min_pt[3] << " " << local_min_pt[4] << " " << local_min_pt[5] << " " << endl;
	
	
	#ifndef NDEBUG_IN
	// Draw the histograms
	int hist_w = 512; int hist_h = 400; // size of figure
	int bin_w = round( (double) hist_w/hist_size ); // width of a bin
	int bin_wh = round( (double) hist_w/hist_size_h );
	Mat hist_image_bgr( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	Mat hist_image_hls( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	
	// find global maximum
	double maxvalhis[6];
	Point mavvalhis_pt[6];
	minMaxLoc(pixhis_b, NULL, &maxvalhis[0], NULL, &mavvalhis_pt[0]);
	minMaxLoc(pixhis_g, NULL, &maxvalhis[1], NULL, &mavvalhis_pt[1]);
	minMaxLoc(pixhis_r, NULL, &maxvalhis[2], NULL, &mavvalhis_pt[2]);
	minMaxLoc(pixhis_h, NULL, &maxvalhis[3], NULL, &mavvalhis_pt[3]);
	minMaxLoc(pixhis_l, NULL, &maxvalhis[4], NULL, &mavvalhis_pt[4]);
	minMaxLoc(pixhis_s, NULL, &maxvalhis[5], NULL, &mavvalhis_pt[5]);
	
	cout << "max loc" << mavvalhis_pt[0] << " " << mavvalhis_pt[1] << " " << mavvalhis_pt[2] << " " << mavvalhis_pt[3] << " " << mavvalhis_pt[4] << " " << mavvalhis_pt[5] << endl;
	cout << "max val" << maxvalhis[0] << " " << maxvalhis[1] << " " << maxvalhis[2] << " " << maxvalhis[3] << " " << maxvalhis[4] << " " << maxvalhis[5] << endl;
	
	// normalize the histogram into the figure
	double max_bgr = max(max(maxvalhis[0], maxvalhis[1]), maxvalhis[2]);
	double max_hls = max(max(maxvalhis[3], maxvalhis[4]), maxvalhis[5]);
	
	pixhis_b = pixhis_b*(hist_h-1)/max_bgr;
	pixhis_g = pixhis_g*(hist_h-1)/max_bgr;
	pixhis_r = pixhis_r*(hist_h-1)/max_bgr;
	pixhis_h = pixhis_h*(hist_h-1)/max_hls;
	pixhis_l = pixhis_l*(hist_h-1)/max_hls;
	pixhis_s = pixhis_s*(hist_h-1)/max_hls;
	
	// Draw for each channel
	for( int i = 1; i < hist_size; i++ )
	{
		line( hist_image_bgr, Point( bin_w*(i-1), hist_h-1 - round(pixhis_b.at<float>(i-1)) ) ,
							  Point( bin_w*(i), hist_h-1 - round(pixhis_b.at<float>(i)) ),
							  Scalar( 255, 0, 0), 2, 8, 0  );
      line( hist_image_bgr, Point( bin_w*(i-1), hist_h-1 - round(pixhis_g.at<float>(i-1)) ) ,
							Point( bin_w*(i), hist_h-1 - round(pixhis_g.at<float>(i)) ),
							Scalar( 0, 255, 0), 2, 8, 0  );
      line( hist_image_bgr, Point( bin_w*(i-1), hist_h-1 - round(pixhis_r.at<float>(i-1)) ) ,
							Point( bin_w*(i), hist_h-1 - round(pixhis_r.at<float>(i)) ),
							Scalar( 0, 0, 255), 2, 8, 0  );
	  line( hist_image_hls, Point( bin_w*(i-1), hist_h-1 - round(pixhis_l.at<float>(i-1)) ) ,
							Point( bin_w*(i), hist_h-1 - round(pixhis_l.at<float>(i)) ),
							Scalar( 0, 255, 0), 2, 8, 0  );
      line( hist_image_hls, Point( bin_w*(i-1), hist_h-1 - round(pixhis_s.at<float>(i-1)) ) ,
							Point( bin_w*(i), hist_h-1 - round(pixhis_s.at<float>(i)) ),
							Scalar( 0, 0, 255), 2, 8, 0  );
	}
	for( int i = 1; i < hist_size_h; i++ )
	{
      line( hist_image_hls, Point( bin_wh*(i-1), hist_h-1 - round(pixhis_h.at<float>(i-1)) ) ,
							Point( bin_wh*(i), hist_h-1 - round(pixhis_h.at<float>(i)) ),
							Scalar( 255, 0, 0), 2, 8, 0  );
	}
  
	line(hist_image_bgr, Point(local_min_pt[0]*bin_w, 0), Point(local_min_pt[0]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_bgr, Point(local_min_pt[1]*bin_w, 0), Point(local_min_pt[1]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_bgr, Point(local_min_pt[2]*bin_w, 0), Point(local_min_pt[2]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
  
	line(hist_image_hls, Point(local_min_pt[3]*bin_wh, 0), Point(local_min_pt[3]*bin_wh, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_hls, Point(local_min_pt[4]*bin_w, 0), Point(local_min_pt[4]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_hls, Point(local_min_pt[5]*bin_w, 0), Point(local_min_pt[5]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
  
	// Display
	imshow("calcHist BGR", hist_image_bgr );
	imshow("calcHist HLS", hist_image_hls );

	waitKey(0);
  
	// test the effect of the threshold
	Mat masked_edges, color_mask;

	Mat height_mask(image.size(), CV_8UC1, Scalar(255));
	#ifndef HIGH_BOT
	height_mask.rowRange(0, image.rows*2/3) = 0;
	#else
	height_mask.rowRange(0, image.rows/2) = 0;
	height_mask.rowRange(image.rows*7/10, image.rows) = 0;  // for caltech data
	#endif
	
	color_mask = bgr_chnl[0] > local_min_pt[0] & bgr_chnl[0] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked b", masked_edges );
	
	color_mask = bgr_chnl[1] > local_min_pt[1] & bgr_chnl[1] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked g", masked_edges );
	
	color_mask = bgr_chnl[2] > local_min_pt[2] & bgr_chnl[2] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked r", masked_edges );
	
	color_mask = hls_chnl[0] > 0 & hls_chnl[0] < local_min_pt[3];
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked h", masked_edges );
	
	color_mask = hls_chnl[1] > local_min_pt[4] & hls_chnl[1] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked l", masked_edges );
	
	color_mask = hls_chnl[2] > local_min_pt[5] & hls_chnl[2] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked s", masked_edges );
	
	Mat blur_edges_bi = blur_edges > 0;
	bitwise_and(image, blur_edges_bi, masked_edges);
	imshow("masked edge", masked_edges );
	waitKey(0);
	
	#endif
	
	return 0;
}