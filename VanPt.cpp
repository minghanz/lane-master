#include "VanPt.h"
#include "LaneImage.hpp"
#include "Line.hpp"
#include "LaneMark.h"

VanPt::VanPt(float alpha_w, float alpha_h) : ALPHA_W(alpha_w), ALPHA_H(alpha_h), chnl_thresh(vector<int>(6, 0)), VAN_TRACK_Y(15.0/720.0*img_size.width)
{
	#ifndef HIGH_BOT
	y_bottom_warp = min(img_size.height * 14/15, img_size.height -1 );
	#else
	y_bottom_warp = min(img_size.height *7/10, img_size.height -1 );  // for caltech data
	#endif

	#ifdef CALI_VAN
	coef_pix_per_cm = 0.0074;
	van_pt_cali_y = 161.1941;		// y = 0.0074(x-161.1941)
	float theta_h_cali = atan(tan(alpha_h)*(1- 2*van_pt_cali_y/img_size.height));

	van_pt_cali = Point2f(img_size.width/2, van_pt_cali_y);
	van_pt_ini = van_pt_cali;

	float real_width_center = 1 / (coef_pix_per_cm*(img_size.height/2 - van_pt_cali_y)) * 60;
	float beta_w = atan(tan(alpha_w)*(2*60/img_size.width)); // 60 is half_width when calibrating van_pt
	float axis_dist = real_width_center / tan(beta_w) ;
	float cam_height = axis_dist * sin(theta_h_cali);

	float real_width_bottom = 1 / (coef_pix_per_cm*(y_bottom_warp - van_pt_cali_y)) * img_size.width;
	warp_pix_per_cm = warp_col / real_width_bottom;
	min_width_pixel_warp = warp_pix_per_cm * 200; // assume lane width should be no less than 200cm

	#else
	van_pt_ini = Point2f(img_size.width/2, img_size.height/2);
	van_pt_cali = van_pt_ini;
	#endif

	
	{
		float y_top_warp = (y_bottom_warp + 5*van_pt_ini.y)/6;
		float x_van = van_pt_ini.x;
		float y_van = van_pt_ini.y;
		float y_bottom = y_bottom_warp;
		float x_left = 0;
		float x_right = img_size.width - 1;
		vector<Point> warp_test;
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_left) + x_left, y_top_warp));
		warp_test.push_back(Point(x_left, y_bottom ));
		warp_test.push_back(Point(x_right, y_bottom ));
		warp_test.push_back(Point((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_right)+ x_right, y_top_warp));
		warp_test_vec.clear();
		warp_test_vec.push_back(warp_test);
	}
	warp_dst.clear();
	warp_dst.push_back(Point2f(warp_col/6, 0 )); 	// remind that calculation with int could result in weird values when division appears!
	warp_dst.push_back(Point2f(warp_col/6, warp_row -1));
	warp_dst.push_back(Point2f(warp_col*5/6, warp_row -1 ));
	warp_dst.push_back(Point2f(warp_col*5/6, 0));
	warp_src.clear();
	warp_src.push_back(Point2f(warp_test_vec[0][0]));
	warp_src.push_back(Point2f(warp_test_vec[0][1]));
	warp_src.push_back(Point2f(warp_test_vec[0][2]));
	warp_src.push_back(Point2f(warp_test_vec[0][3]));

	per_mtx = getPerspectiveTransform(warp_src, warp_dst);
	inv_per_mtx = getPerspectiveTransform(warp_dst, warp_src);

	first_sucs = false;
    sucs_before = false;
    pos_of_renew_van = 0;
	fail_ini_count = 0;
	
	consist = true;
}

void VanPt::checkClearSeriesVan()
{
	if (fail_ini_count >= 20)
	{
		series_van_pt.clear();
		pos_of_renew_van = 0;
		fail_ini_count = 0;
	}
}

int VanPt::initialVan(Mat color_img, Mat image) // renewing van_pt_ini, y_bottom_warp, warp_test_vec, warp_src, fail_ini_count, chnl_thresh
{	
	first_sucs = false;

	/// Canny and edge filter
	Mat edges;
	Canny(image, edges, 50, 150, 3);
	
	#ifndef NDEBUG_IN
	imshow("canny", edges);
	#endif
	
	

	Mat sobel_x, sobel_y, sobel_angle;
	Sobel(image, sobel_x, CV_32F, 1, 0, 7); // 3
	Sobel(image, sobel_y, CV_32F, 0, 1, 7);
	phase(sobel_x, sobel_y, sobel_angle, true);  // output of phase in degree is [0~360]
	Mat angle_mask;
	angle_mask = (sobel_angle >= 10 & sobel_angle <= 80) | (sobel_angle >= 100 & sobel_angle <= 170) | (sobel_angle >= 190 & sobel_angle <= 260) | (sobel_angle >= 280 & sobel_angle <= 350);

	
	bitwise_and(edges, angle_mask, edges); // remove edges with wrong angle
	
	Mat cali_mask;
	Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(7, 7) );
	erode(image, cali_mask, erode_kernel );
	cali_mask = cali_mask > 0;
	bitwise_and(edges, cali_mask, edges); // remove the edges caused by warp effect


	Mat steer_resp_mag(image.size(), CV_32FC1, Scalar(0));
	Mat steer_angle_max(image.size(), CV_32FC1, Scalar(0));
	Mat steer_resp_weight(image.size(), CV_32FC1, Scalar(0));
	SteerFilter(image, steer_resp_mag, steer_angle_max, steer_resp_weight);
    
    Mat gabor_resp_dir(image.size(), CV_32FC1, Scalar(0));
    Mat gabor_resp_mag(image.size(), CV_32FC1, Scalar(0));
    Mat gabor_weight(image.size(), CV_32FC1, Scalar(0));
    Mat gabor_vote(image.size(), CV_32FC1, Scalar(0));
    GaborFilter(image, gabor_resp_mag, gabor_resp_dir, gabor_weight);
    cout << "Ok 3" << endl;
    GaborVote(gabor_resp_dir, gabor_weight, gabor_vote);
    // GaborVote(steer_angle_max, steer_resp_weight, gabor_vote);
    
    if (sucs_before)
	{
		edges.rowRange(0, warp_test_vec[0][0].y) = 0;
		edges.rowRange(warp_test_vec[0][1].y, image.rows) = 0;
		
	}

    Mat blur_edges;
	LaneDirec(gabor_weight, edges, blur_edges); // steer_resp_mag
	
	
	////////////////////// need to be added to LaneDirec
	// #ifndef NDEBUG_IN   // draw van_pt_avg tracking rectangle
	// Mat edges_w_hist_van;
	// edges.copyTo(edges_w_hist_van);
	// if (series_van_pt.size() == 5) 
	// 	rectangle(edges_w_hist_van, Point(image.cols/4, van_pt_avg.y - VAN_TRACK_Y*2), Point(image.cols*3/4, van_pt_avg.y + VAN_TRACK_Y*2), Scalar(255), -1);
	// imshow("masked canny", edges_w_hist_van);
	// #endif
	/////////////////////////////////////////////////////////
	
    DecideChnlThresh(color_img, image, blur_edges); // use normalized one or ?


	
	/// generate the trapezoid for warping and masking
	float y_top_warp = (y_bottom_warp + 5*van_pt_ini.y)/6;
	float x_van = van_pt_ini.x;
	float y_van = van_pt_ini.y;
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

	warp_src.clear();
	warp_src.push_back(Point2f(warp_test_vec[0][0] )); 	// remind that calculation with int could result in weird values when division appears!
	warp_src.push_back(Point2f(warp_test_vec[0][1] ));
	warp_src.push_back(Point2f(warp_test_vec[0][2] ));
	warp_src.push_back(Point2f(warp_test_vec[0][3] ));

	per_mtx = getPerspectiveTransform(warp_src, warp_dst);
	inv_per_mtx = getPerspectiveTransform(warp_dst, warp_src);


	#ifdef CALI_VAN
	float real_width_bottom = 1 / (coef_pix_per_cm*(y_bottom_warp - van_pt_cali_y)) * img_size.width;
	warp_pix_per_cm = warp_col / real_width_bottom;
	min_width_pixel_warp = warp_pix_per_cm * 200; // assume lane width should be no less than 200cm
	#endif
	
	
	
	if (!sucs_before)
	{
		sucs_before = true;
		first_sucs = true;
	}

	return 0;
}

void localMaxima(const Mat& src, int num_nonz, int& min_loc, int hist_size)
{
	float accu_num_pix = 0;
	//float total_pix = image.rows * image.cols/4 ;
	int i;
	for (i = 0; i < hist_size; i++)
	{
		accu_num_pix += src.at<float>(hist_size - 1 - i);
		if (accu_num_pix >= 0.5 * num_nonz)
			break;
	}
	min_loc = hist_size - 1 - i;
	
	return;
}


void VanPt::recordHistVan(LaneImage& lane_find_image, Line& left_lane, Line& right_lane)
{
	van_pt_img = lane_find_image.__van_pt;
	bool detected = left_lane.detected && right_lane.detected;
	Point new_van_pt = van_pt_img;

	//float size_ref = img_size.width > img_size.height ? img_size.height : img_size.width;
	if ( series_van_pt.empty() )
	{
		series_van_pt.push_back(new_van_pt);
		return;
	}
	else 
	{
		vector<Point>::iterator first = series_van_pt.begin();
		vector<Point>::iterator last = series_van_pt.end();
		if (last - first == 1)
		{
			van_pt_avg = *first;
			if ( ( (abs(new_van_pt.x - first->x) <= 2*VAN_TRACK_Y) || (abs(new_van_pt.x - img_size.width/2) < abs(first->x - img_size.width/2)) ) && abs(new_van_pt.y - first->y) <= VAN_TRACK_Y)
				series_van_pt.push_back(new_van_pt);
			else
				*first = new_van_pt;
		}
		else if (last - first < 5)
		{
			float x_avg = 0, y_avg = 0;
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				x_avg += it->x;
				y_avg += it->y;
			}
			x_avg = x_avg / (last-first);
			y_avg = y_avg / (last-first);
			van_pt_avg.x = (int)x_avg;
			van_pt_avg.y = (int)y_avg;
			
			if ( ( abs(new_van_pt.x - x_avg) <= 2*VAN_TRACK_Y || (abs(new_van_pt.x - img_size.width/2) < abs(x_avg - img_size.width/2)) ) && abs(new_van_pt.y - y_avg) <= VAN_TRACK_Y)
				series_van_pt.push_back(new_van_pt);
			else if ( !detected ) // this step is before Line::processNewRecord, so it refers to detected in last frame
			{
				series_van_pt.clear();
				series_van_pt.push_back(new_van_pt);
				pos_of_renew_van = 0;
			}

		}
		else if (last - first == 5)
		{
			float x_avg = 0, y_avg = 0;
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				x_avg += it->x;
				y_avg += it->y;
			}
			x_avg = x_avg / (last-first);
			y_avg = y_avg / (last-first);
			van_pt_avg.x = (int)x_avg;
			van_pt_avg.y = (int)y_avg;
			
			if ( ( abs(new_van_pt.x - x_avg) <= 2*VAN_TRACK_Y  || (abs(new_van_pt.x - img_size.width/2) < abs(x_avg - img_size.width/2)) ) && abs(new_van_pt.y - y_avg) <= VAN_TRACK_Y) // temporally not using the relation with last point
			{
				consist = true;
				series_van_pt[pos_of_renew_van] = new_van_pt;
				pos_of_renew_van = (pos_of_renew_van + 1) % 5;
			}
			else if ( !detected )
			{
				consist = true;
				series_van_pt.clear();
				series_van_pt.push_back(new_van_pt);
				pos_of_renew_van = 0;
			}
			else
				consist = false;
			
			cout << "pos_of_renew_van: " << pos_of_renew_van << ", consist: " << consist << endl;
		}
		else
		{
			cout << "Wrong number of points in the series. " << endl;
			while (series_van_pt.end() - first != 5)
				series_van_pt.pop_back();
			cout << "Now the length of point series: " << series_van_pt.end() - series_van_pt.begin() << endl;
		}
		return;
	}
}

void VanPt::recordBestVan(Line& left_lane, Line& right_lane)
{
	if (left_lane.detected && right_lane.detected)
	{
		float w_current = (left_lane.__w_current) < (right_lane.__w_current) ? (left_lane.__w_current) : (right_lane.__w_current) ;
		van_pt_best = w_current*van_pt_img + (1-w_current)*van_pt_best;
		van_pt_best_int = Point(van_pt_best);
	
		theta_w = atan(tan(ALPHA_W)*(1- 2*van_pt_best.x/img_size.width))*180/CV_PI;
		theta_h = atan(tan(ALPHA_H)*(1- 2*van_pt_best.y/img_size.height))*180/CV_PI;
	}
	return;
}

void VanPt::drawOn(Mat& newwarp, LaneMark& lane_mark)
{
	bool initial_frame = lane_mark.initial_frame;
	bool new_result = lane_mark.new_result;
	cout << "initial_frame: " << initial_frame << endl;
	cout << "new_result: " << new_result<< endl;
	cout << "ini_flag: " << ini_flag<< endl;
	
	
	if (new_result)
	{
		circle(newwarp, van_pt_best_int, 5, Scalar(0, 0, 255), -1);
	}
	
	rectangle(newwarp, Point(van_pt_avg) - Point(VAN_TRACK_Y*2, VAN_TRACK_Y), Point(van_pt_avg) + Point(VAN_TRACK_Y*2, VAN_TRACK_Y), Scalar(0, 0, 255), 2);

	if (initial_frame)
	{
		if (series_van_pt.size() == 5)
		{
		rectangle(newwarp, Point(van_pt_avg) - Point(VAN_TRACK_Y*4, VAN_TRACK_Y*2), Point(van_pt_avg) + Point(VAN_TRACK_Y*4, VAN_TRACK_Y*2), Scalar(0, 0, 255), 2);
		}
		if (ini_flag ==0)
		{

			circle(newwarp, Point(van_pt_ini), 5, Scalar(0, 200, 150), -1); // van_pt for the initial perspective transform
			cout << "drawed" << endl;
		}
	}
		

	vector<Point> warp_src_int;
	warp_src_int.push_back(Point(warp_src[0]));
	warp_src_int.push_back(Point(warp_src[1]));
	warp_src_int.push_back(Point(warp_src[2]));
	warp_src_int.push_back(Point(warp_src[3]));
	warp_test_vec.clear();
	warp_test_vec.push_back(warp_src_int);
	drawContours(newwarp, warp_test_vec, -1, Scalar(255, 0, 0), 5 );
	#ifndef NDEBUG
	cout << " current warp vertex: " << warp_test_vec[0][0] << warp_test_vec[0][1] << warp_test_vec[0][2] << warp_test_vec[0][3] << endl;
	#endif
}

void VanPt::renewWarp()
{
	warp_src.clear();
	float x_van = van_pt_best.x;
	float y_van = van_pt_best.y;
	float y_bottom = y_bottom_warp;
	float y_top_warp = (y_bottom_warp + 5*y_van)/6;
	float x_left = 0;
	float x_right = img_size.width - 1;
	#ifdef COUT	
	cout << "y_bottom_warp: " << y_bottom_warp << endl;
	#endif
	warp_src.push_back(Point2f((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_left) + x_left, y_top_warp)); 	// remind that calculation with int could result in weird values when division appears!
	warp_src.push_back(Point2f(x_left, y_bottom ));
	warp_src.push_back(Point2f(x_right, y_bottom ));
	warp_src.push_back(Point2f((y_top_warp-y_bottom)/(y_van-y_bottom)*(x_van-x_right)+ x_right, y_top_warp));

	per_mtx = getPerspectiveTransform(warp_src, warp_dst);
	inv_per_mtx = getPerspectiveTransform(warp_dst, warp_src);
}

void VanPt::getSteerKernel(Mat& kernel_x, Mat& kernel_y, Mat& kernel_xy, int ksize, double sigma)
{
	if (ksize % 2 == 0)
		ksize = ksize + 1;
	int center = (ksize + 1)/2;
	
	Mat coord_x(ksize, ksize, CV_32FC1);
	Mat coord_y(ksize, ksize, CV_32FC1);
	for (int i = 1; i <= ksize; i++)
	{
		coord_x.col(i-1) = i - center;
		coord_y.row(i-1) = i - center; 
	}


	Mat exp_xy, x_sqr, y_sqr;
	pow(coord_x, 2, x_sqr);
	pow(coord_y, 2, y_sqr);
	// Mat sqr_norm = -(x_sqr + y_sqr)/(sigma*sigma);
	exp( -(x_sqr + y_sqr)/(sigma*sigma) , exp_xy);

	kernel_x = ( -(2*x_sqr/(sigma*sigma) - 1)*2/(sigma*sigma) ).mul(exp_xy);
	kernel_y = ( -(2*y_sqr/(sigma*sigma) - 1)*2/(sigma*sigma) ).mul(exp_xy);
	kernel_xy = ( 4*coord_x.mul(coord_y)/(sigma*sigma*sigma*sigma) ).mul(exp_xy);

}

void VanPt::SteerFilter(Mat image, Mat& steer_resp_mag, Mat& steer_angle_max, Mat& steer_resp_weight)
{
	int ksize = 21;
	double sigma = 4; // 8

	Mat kernel_steer_x(ksize, ksize, CV_32F);
	Mat kernel_steer_y(ksize, ksize, CV_32F);
	Mat kernel_steer_xy(ksize, ksize, CV_32F);
	cout <<"lalala 0 " << endl;
	getSteerKernel(kernel_steer_x, kernel_steer_y, kernel_steer_xy, ksize, sigma);
	cout <<"lalala 1 " << endl;
	Mat steer_resp_x, steer_resp_y, steer_resp_xy;
	filter2D(image, steer_resp_x, CV_32F, kernel_steer_x );
	filter2D(image, steer_resp_y, CV_32F, kernel_steer_y );
	filter2D(image, steer_resp_xy, CV_32F, kernel_steer_xy );


	Mat steer_resp_max_x(image.size(), CV_32FC1, Scalar(0));
	Mat steer_resp_max_y(image.size(), CV_32FC1, Scalar(0));
	Mat steer_resp_max_xy(image.size(), CV_32FC1, Scalar(0));

	Mat steer_resp_min_x(image.size(), CV_32FC1, Scalar(0));
	Mat steer_resp_min_y(image.size(), CV_32FC1, Scalar(0));
	Mat steer_resp_min_xy(image.size(), CV_32FC1, Scalar(0));


	// sobel_angle = sobel_angle *CV_PI/180 - CV_PI/2;
	// Mat A;
	// Mat steer_resp_x_2, steer_resp_y_2;
	// pow(steer_resp_x, 2, steer_resp_x_2);
	// pow(steer_resp_y, 2, steer_resp_y_2);
	// sqrt(steer_resp_x_2 - 2*steer_resp_x.mul(steer_resp_y)+steer_resp_y_2 + 4*steer_resp_xy, A);
	// Mat steer_angle_tan_max = (steer_resp_x - steer_resp_y + A )/(2*steer_resp_xy);
	// Mat steer_angle_tan_min = (steer_resp_x - steer_resp_y - A )/(2*steer_resp_xy);
	
	// Mat steer_angle_max(image.size(), CV_32FC1);
	Mat steer_angle_min(image.size(), CV_32FC1);

	Mat steer_angle_max_cos_2(image.size(), CV_32FC1);
	Mat steer_angle_max_sin_2(image.size(), CV_32FC1);
	
	// cout << sobel_angle.depth() << endl;
	for (int i = image.rows/2; i < y_bottom_warp; i ++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			// steer_angle_max.at<float>(i,j) = atan(steer_angle_tan_max.at<float>(i,j));
			// steer_angle_max_cos_2.at<float>(i,j) = cos(steer_angle_max.at<float>(i,j))*cos(steer_angle_max.at<float>(i,j));
			// steer_angle_max_sin_2.at<float>(i,j) = sin(steer_angle_max.at<float>(i,j))*sin(steer_angle_max.at<float>(i,j));
			float Ixx = steer_resp_x.at<float>(i,j);
			float Iyy = steer_resp_y.at<float>(i,j);
			float Ixy = steer_resp_xy.at<float>(i,j);
			
			if (Ixx > Iyy)
			{
				steer_angle_max.at<float>(i,j) = 0.5*atan(Ixy/(Ixx-Iyy));
			}
			else if (Ixx < Iyy)
			{
				steer_angle_max.at<float>(i,j) = 0.5*(CV_PI+atan(Ixy/(Ixx-Iyy)));
			}
			else
			{
				steer_angle_max.at<float>(i,j) = CV_PI/4;
			}
			steer_angle_min.at<float>(i,j) = steer_angle_max.at<float>(i,j) + CV_PI/2;
			
			steer_resp_max_x.at<float>(i,j) = steer_resp_x.at<float>(i,j)*cos(steer_angle_max.at<float>(i,j))*cos(steer_angle_max.at<float>(i,j)); //*abs(cos(steer_angle_max.at<float>(i,j)))
			steer_resp_max_y.at<float>(i,j) = steer_resp_y.at<float>(i,j)*sin(steer_angle_max.at<float>(i,j))*sin(steer_angle_max.at<float>(i,j));
			steer_resp_max_xy.at<float>(i,j) = steer_resp_xy.at<float>(i,j)*sin(steer_angle_max.at<float>(i,j))*cos(steer_angle_max.at<float>(i,j));
		
			// steer_angle_min.at<float>(i,j) = atan(steer_angle_tan_min.at<float>(i,j));
			steer_resp_min_x.at<float>(i,j) = steer_resp_x.at<float>(i,j)*cos(steer_angle_min.at<float>(i,j))*cos(steer_angle_min.at<float>(i,j));
			steer_resp_min_y.at<float>(i,j) = steer_resp_y.at<float>(i,j)*sin(steer_angle_min.at<float>(i,j))*sin(steer_angle_min.at<float>(i,j));
			steer_resp_min_xy.at<float>(i,j) = steer_resp_xy.at<float>(i,j)*sin(steer_angle_min.at<float>(i,j))*cos(steer_angle_min.at<float>(i,j));
		
			float angle_2_center = atan((van_pt_cali.x - j)/( i - van_pt_cali.y ) ); 
			steer_resp_weight.at<float>(i,j) =abs(cos(steer_angle_max.at<float>(i,j)) *cos(angle_2_center + steer_angle_max.at<float>(i,j))*(float)(i-image.rows/2)/(y_bottom_warp-image.rows/2));
		}
	}
	Mat steer_resp_max = steer_resp_max_x + steer_resp_max_y + 2*steer_resp_max_xy;
	Mat steer_resp_min = steer_resp_min_x + steer_resp_min_y + 2*steer_resp_min_xy;
	
	steer_resp_mag = steer_resp_max - steer_resp_min;
	steer_resp_weight = steer_resp_weight.mul(steer_resp_mag);

	steer_angle_max = -steer_angle_max;
	for (int j = 217; j < 227; j++)
	{
		cout << "angle at (317, " << j << "): " <<  steer_angle_max.at<float>(317,j)*180/CV_PI << endl;
		cout << "mag at (317, " << j << "): " <<  steer_resp_mag.at<float>(317,j)*180/CV_PI << endl;
		cout << "x at (317, " << j << "): " <<  steer_resp_x.at<float>(317,j)*180/CV_PI << endl;
		cout << "y at (317, " << j << "): " <<  steer_resp_y.at<float>(317,j)*180/CV_PI << endl;
		cout << "xy at (317, " << j << "): " <<  steer_resp_xy.at<float>(317,j)*180/CV_PI << endl;
		
	}

	cout <<"lalala 2 " << endl;
	Mat steer_resp_x_show, steer_resp_y_show, steer_resp_xy_show, steer_resp_mag_show, steer_resp_weight_show;
	

	// cout << "steer_angle_max_cos_2 at (10,464): " << steer_angle_max_cos_2.at<float>(464,10) <<  endl;
	// cout << "steer_angle_max_sin_2 at (10,464): " << steer_angle_max_sin_2.at<float>(464,10) <<  endl;
	// cout << "steer_angle_tan_max at (10,464): " << steer_angle_tan_max.at<float>(464,10) <<  endl;
	cout << "steer_angle_max at (10,464): " << steer_angle_max.at<float>(464,10) <<  endl;
	cout << "steer_resp_xy at (10,464): " << steer_resp_xy.at<float>(464,10) <<  endl;
	cout << "steer_resp_x at (10,464): " << steer_resp_x.at<float>(464,10) <<  endl;
	cout << "steer_resp_y at (10,464): " << steer_resp_y.at<float>(464,10) <<  endl;
	// cout << "A at (10,464): " << A.at<float>(464,10) <<  endl;
	// cout << "result at (10,464)" << (steer_resp_x.at<float>(464,10)  -steer_resp_y.at<float>(464,10) + A.at<float>(464,10)) / (2*steer_resp_xy.at<float>(464,10)) << endl;
	
	
	// normalize(steer_angle_max_cos_2, steer_angle_max_cos_2, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_angle_max_sin_2, steer_angle_max_sin_2, 0, 255, NORM_MINMAX , CV_8U);
	// imshow("steer_angle_max_cos_2", steer_angle_max_cos_2);
	// imshow("steer_angle_max_sin_2", steer_angle_max_sin_2);


	normalize(steer_resp_mag, steer_resp_mag_show, 0, 255, NORM_MINMAX , CV_8U);
	normalize(steer_resp_weight, steer_resp_weight_show, 0, 255, NORM_MINMAX , CV_8U);
	normalize(steer_resp_max, steer_resp_max, 0, 255, NORM_MINMAX , CV_8U);
	normalize(steer_resp_min, steer_resp_min, 0, 255, NORM_MINMAX , CV_8U);
	imshow("steer_resp_max", steer_resp_max);
	imshow("steer_resp_min", steer_resp_min);
	imshow("steer_resp_mag", steer_resp_mag_show);
	imshow("steer_resp_weight", steer_resp_weight_show);
	
	

	// Mat steer_resp_max_x_show, steer_resp_max_y_show, steer_resp_max_xy_show;
	// Mat steer_resp_min_x_show, steer_resp_min_y_show, steer_resp_min_xy_show;
	// normalize(steer_resp_max_x, steer_resp_max_x_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_max_y, steer_resp_max_y_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_max_xy, steer_resp_max_xy_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_min_x, steer_resp_min_x_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_min_y, steer_resp_min_y_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_min_xy, steer_resp_min_xy_show, 0, 255, NORM_MINMAX , CV_8U);

	// imshow("steer_resp_max_x_show", steer_resp_max_x_show);
	// imshow("steer_resp_max_y_show", steer_resp_max_y_show);
	// imshow("steer_resp_max_xy_show", steer_resp_max_xy_show);
	// imshow("steer_resp_min_x_show", steer_resp_min_x_show);
	// imshow("steer_resp_min_y_show", steer_resp_min_y_show);
	// imshow("steer_resp_min_xy_show", steer_resp_min_xy_show);

	// normalize(steer_resp_x, steer_resp_x_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_y, steer_resp_y_show, 0, 255, NORM_MINMAX , CV_8U);
	// normalize(steer_resp_xy, steer_resp_xy_show, 0, 255, NORM_MINMAX , CV_8U);
	// imshow("steer_resp_x_show", steer_resp_x_show);
	// imshow("steer_resp_y_show", steer_resp_y_show);
	// imshow("steer_resp_xy_show", steer_resp_xy_show);

	



	Mat kernel_steer_x_show, kernel_steer_y_show, kernel_steer_xy_show;
	normalize(kernel_steer_x, kernel_steer_x_show, 0, 255, NORM_MINMAX , CV_8U);
	normalize(kernel_steer_y, kernel_steer_y_show, 0, 255, NORM_MINMAX , CV_8U);
	normalize(kernel_steer_xy, kernel_steer_xy_show, 0, 255, NORM_MINMAX , CV_8U);
	
	imshow("steer_kern_x", kernel_steer_x_show);
	imshow("steer_kern_y", kernel_steer_y_show);
	imshow("steer_kern_xy", kernel_steer_xy_show);

	

	
	waitKey(0);
}


void VanPt::GaborFilter(Mat image, Mat& gabor_resp_mag, Mat& gabor_resp_dir, Mat& gabor_weight)
{
	Mat gaborKernelReal[4];
	Mat gaborKernelImage[4];
	Mat gabor_resp_real[4];
	Mat gabor_resp_image[4];
	Mat gabor_resp_engy[4];
	// Mat gabor_resp_mag(image.size(), CV_32F, Scalar(0));
	// Mat gabor_resp_dir(image.size(), CV_32F, Scalar(0));
	// Mat gabor_weight(image.size(), CV_32F, Scalar(0));

    double lambda = 16; // 4*sqrt(2)
    double radial_frequency = 2*CV_PI/lambda;
    double K = CV_PI/2;
    double sigma = K/radial_frequency;
    double gamma = 0.5;
    int ksize = ((int)(sigma*9))/2*2+1;
    double psi=CV_PI*0;
	int ktype=CV_32F;
	cout << "Ok 1" << endl;
    for (int i = 0; i < 4; i++)
    {
        double theta = i*CV_PI/4;
        gaborKernelReal[i] = getGaborKernel(Size(ksize,ksize), sigma, theta, lambda, gamma, psi, ktype);
		gaborKernelImage[i] = getGaborKernel(Size(ksize,ksize), sigma, theta, lambda, gamma, psi-CV_PI/2, ktype); // cos(x - pi/2) = sin(x)
		
		stringstream ss;
		ss << "gabor_kernel " << i;
		string image_name = ss.str();
		Mat show_garbor;
		normalize(gaborKernelReal[i], show_garbor, 0, 255, NORM_MINMAX, CV_8U);
		imshow(image_name, show_garbor);
		normalize(gaborKernelImage[i], show_garbor, 0, 255, NORM_MINMAX, CV_8U);
		imshow(image_name+" im", show_garbor);
		filter2D(image, gabor_resp_real[i], CV_32F, gaborKernelReal[i]);
		filter2D(image, gabor_resp_image[i], CV_32F, gaborKernelImage[i]);
		
		sqrt(gabor_resp_real[i].mul(gabor_resp_real[i])+gabor_resp_image[i].mul(gabor_resp_image[i]), gabor_resp_engy[i] );
	}
	
	for (int i = image.rows/2; i < y_bottom_warp; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Vec4f resp_engy(gabor_resp_engy[0].at<float>(i,j), gabor_resp_engy[1].at<float>(i,j), gabor_resp_engy[2].at<float>(i,j), gabor_resp_engy[3].at<float>(i,j) ); 
			
			Vec4i sort_idx;
			sortIdx(resp_engy, sort_idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING );
			
			float v1 = resp_engy[sort_idx[0]] - resp_engy[sort_idx[3]];
			float v2 = resp_engy[sort_idx[1]] - resp_engy[sort_idx[2]];
            
			float x_proj = v1*sin(sort_idx[0]*CV_PI/4) + v2*sin(sort_idx[1]*CV_PI/4);
			float y_proj = v1*cos(sort_idx[0]*CV_PI/4) + v2*cos(sort_idx[1]*CV_PI/4);
			gabor_resp_mag.at<float>(i,j) = sqrt( x_proj*x_proj + y_proj*y_proj );
			gabor_resp_dir.at<float>(i,j) = atan(x_proj/y_proj) + (y_proj<0)*CV_PI ;
            float angle_to_last_van = atan((van_pt_cali.x - j)/( i - van_pt_cali.y ) ); 
            if (i == 556 && j > 478 && j < 492) // 456-492
			{
				cout << "resp_descend: " << i<< " " << j << " " << resp_engy[0] << " " << resp_engy[1] << " " << resp_engy[2] << " " << resp_engy[3]<< endl; 
                cout << "resp_descend: " << i<< " " << j << " " << gabor_resp_dir.at<float>(i,j)*180/CV_PI << " " << angle_to_last_van*180/CV_PI << endl;
            }
			gabor_weight.at<float>(i,j) = gabor_resp_mag.at<float>(i,j)* abs(cos( angle_to_last_van-gabor_resp_dir.at<float>(i,j) ) * cos(gabor_resp_dir.at<float>(i,j)))*abs((float)(i-image.rows/2)/(y_bottom_warp-image.rows/2)); //* (v1 > resp_engy[sort_idx[3]])
			// weighted by angle to van_cali, angle to perpendicular, distance to bottom
		}
	}
	cout << "Ok 4" << endl;
	Mat show_garbor;
	normalize(gabor_weight, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
    imshow("gabor_weight", show_garbor);
    normalize(gabor_resp_mag, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
    imshow("gabor_resp_mag", show_garbor);
    
	waitKey(0);
}

void VanPt::GaborVote(Mat gabor_resp_dir, Mat gabor_weight, Mat& vote_map)
{
	// first transform it with Hough
	Mat gabor_weight_8U;
	normalize(gabor_weight, gabor_weight_8U, 0, 255, NORM_MINMAX, CV_8U);
	double thresh = 50;
	threshold(gabor_weight_8U, gabor_weight_8U, thresh, 0, THRESH_TOZERO);
	Size ksize(15,5);
	double sigmaX = 7;
	double sigmaY = 2;
	GaussianBlur(gabor_weight_8U, gabor_weight_8U, ksize, sigmaX, sigmaY );
	Mat gabor_weight_8U_nms(gabor_resp_dir.size(), CV_8UC1, Scalar(0));
	NMS(gabor_weight_8U, gabor_weight_8U_nms);
	Mat dilate_kern = getStructuringElement(MORPH_RECT, Size(2,2));
	dilate(gabor_weight_8U_nms, gabor_weight_8U_nms, dilate_kern); 
	imshow("gabor_weight_CV_8U", gabor_weight_8U);
	imshow("gabor_weight_CV_8U_nms", gabor_weight_8U_nms);

	vector<Vec4i> lines;
	double rho = 20;
	double theta = 3*CV_PI/180;
	int threshol = 30;
	double minLineLength = 30;
	double maxLineGap = 10;
	HoughLinesP(gabor_weight_8U_nms, lines, rho, theta, threshol, minLineLength, maxLineGap );
	// HoughLines(gabor_weight_8U_nms, lines, rho, theta, threshol, 0, 0, CV_PI/12, CV_PI*11/12 );
	cout << "hough is good" << endl;
	Mat vote_line_ori(gabor_resp_dir.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < lines.size(); i++)
	{
		line(vote_line_ori, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255) );
	}
	imshow("vote_line_ori",vote_line_ori);
		

	Mat vote_left(gabor_resp_dir.size(), CV_32FC1, Scalar(0));
	Mat vote_right(gabor_resp_dir.size(), CV_32FC1, Scalar(0));
	Mat vote_line(gabor_resp_dir.size(), CV_32FC1, Scalar(0));
	float y_bottom_left = 0, y_bottom_right= 0;
	for (int i = 0; i < lines.size(); i++)
	{
		float x1 = lines[i][0];
		float y1 = lines[i][1];
		float x2 = lines[i][2];
		float y2 = lines[i][3];
		double w = 0; //(abs(x1-x2) +abs(y1-y2)); // accumulative weight
		float k = (x1-x2)/(y1-y2);
		
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
			if (x0 < img_size.width / 2)
				continue;
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > img_size.width - 1 || x_cur < 0)
					break;
				// w += gabor_weight.at<float>(y_cur, x_cur);
				w += gabor_weight_8U.at<uchar>(y_cur, x_cur);
				vote_right.at<float>(y_cur, x_cur)+= w;
			}
			if (x0 + k*(y_bottom_warp - y0)> img_size.width - 1) // not approaching bottom
			{
				float lower_y = y0 + (img_size.width - 1 - x0)/k;
				if (lower_y > y_bottom_right)
					y_bottom_right = lower_y;
			}
			else
				y_bottom_right = y_bottom_warp;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
		}
		else // left
		{
			if (x0 > img_size.width / 2)
				continue;
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > img_size.width - 1 || x_cur < 0)
					break;
				// w += gabor_weight.at<float>(y_cur, x_cur);
				w += gabor_weight_8U.at<uchar>(y_cur, x_cur);
				vote_left.at<float>(y_cur, x_cur)+= w;
			}
			if (x0 + k*(y_bottom_warp - y0)< 0) // not approaching bottom
			{
				float lower_y = y0 + (0 - x0)/k;
				if (lower_y > y_bottom_left)
					y_bottom_left = lower_y;
			}
			else
				y_bottom_left = y_bottom_warp;
			line(vote_line, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255), 1);
		}
	}

	if ((y_bottom_left != y_bottom_warp || y_bottom_right != y_bottom_warp ) && y_bottom_left != 0 && y_bottom_right != 0)
	{
		y_bottom_warp = min(y_bottom_left, y_bottom_right);
	}

	// vote_left and right need blurring, because the line is not strictly continuous 
	GaussianBlur(vote_left, vote_left, Size(3,3), 1.5, 1.5 );
	GaussianBlur(vote_right, vote_right, Size(3,3), 1.5, 1.5 );
	
	

	vote_map = 2*vote_left.mul(vote_right)/(vote_left + vote_right);
	Point van_pt_candi;
	double maxval;
	minMaxLoc(vote_map, NULL, &maxval, NULL, &van_pt_candi);
	if (maxval > 0)
    {
		van_pt_ini = Point2f(van_pt_candi);
		ini_success = true;
		cout << "maxval of vote: " << maxval << endl;
	}
	else
	{
		ini_success = false;
		cout << "maxval of vote: " << maxval << endl;
	}
	int thickness = ini_success ? -1:2;
	Mat show_garbor;
	normalize(vote_left+vote_right, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	circle(show_garbor, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("vote_left_right", show_garbor);
	imshow("vote_left", vote_left);
	imshow("vote_right", vote_right);
	normalize(gabor_weight, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	circle(show_garbor, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("gabor_weight", show_garbor);
	circle(vote_line, Point(van_pt_ini), 5, Scalar( 255), thickness);
	imshow("vote_line", vote_line);
	waitKey(0);


	// Mat gabor_vote_l(gabor_resp_dir.size(), CV_32FC1, Scalar(0));
	// Mat gabor_vote_r(gabor_resp_dir.size(), CV_32FC1, Scalar(0));
	// for (int i = gabor_resp_dir.rows/2; i < y_bottom_warp; i+=3)
	// {
	// 	for (int j = 0; j < gabor_resp_dir.cols; j+=3)
	// 	{
	// 		float cur_weight = gabor_weight.at<float>(i,j);
	// 		if (cur_weight <= 0)
	// 		{
	// 			continue;
	// 		}
	// 		float tan_angle = tan(gabor_resp_dir.at<float>(i, j));
	// 		if (tan_angle > 0)
	// 		{
	// 			for (int y = i - 1; y > gabor_resp_dir.rows / 4; y-- )
	// 			{
	// 				int x = j + (i - y) * tan_angle;
	// 				if (x >= gabor_resp_dir.cols || x < 0)
	// 				{
	// 					break;
	// 				}
	// 				gabor_vote_l.at<float>(y, x) += cur_weight;//*(1-(float)(i-y)/(i-gabor_resp_dir.rows/4)*(i-y)/(i-gabor_resp_dir.rows/4));
	// 			}
	// 		}
	// 		else
	// 		{
	// 			for (int y = i - 1; y > gabor_resp_dir.rows / 4; y-- )
	// 			{
	// 				int x = j + (i - y) * tan_angle;
	// 				if (x >= gabor_resp_dir.cols || x < 0)
	// 				{
	// 					break;
	// 				}
	// 				gabor_vote_r.at<float>(y, x) += cur_weight;//*(1-(float)(i-y)/(i-gabor_resp_dir.rows/4)*(i-y)/(i-gabor_resp_dir.rows/4));
	// 			}
	// 		}
			
	// 	}
	// }
	// gabor_vote = 2*gabor_vote_l.mul(gabor_vote_r)/(gabor_vote_l + gabor_vote_r);
	// Point van_pt_candi;
	// minMaxLoc(gabor_vote, NULL, NULL, NULL, &van_pt_candi);
	// van_pt_ini = Point2f(van_pt_candi);

	// Mat show_garbor;
	// normalize(gabor_vote, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	// circle(show_garbor, van_pt_candi, 5, Scalar( 255), -1);
	// imshow("gabor_vote", show_garbor);
	// normalize(gabor_weight, show_garbor, 0, 255, NORM_MINMAX, CV_8U);
	// circle(show_garbor, van_pt_candi, 5, Scalar( 255), -1);
	// imshow("gabor_weight", show_garbor);
	// waitKey(0);
}

void VanPt::LaneDirec(Mat steer_resp_mag, Mat edges, Mat& blur_edges)
{
	Mat mask_left, mask_right; 
	
	RoughLaneDirec(steer_resp_mag, mask_left, 1);
	RoughLaneDirec(steer_resp_mag, mask_right, -1);

	Mat mask_both_sides = mask_left | mask_right;
	imshow("mask_by_steer", mask_both_sides);
	blur_edges = edges & mask_both_sides;
	Mat dilate_kernel = getStructuringElement(MORPH_RECT, Size(3,3) ); // 6*6 for big image, 3*3 for small image
	dilate(blur_edges, blur_edges, dilate_kernel);
	imshow("blur_edges", blur_edges);
	waitKey(0);

}

void VanPt::RoughLaneDirec(Mat steer_resp_mag, Mat& mask_side, int direction)
{
	mask_side = Mat(img_size, CV_8UC1, Scalar(0));
	float max_sum_left_steer = 0;
	
	int max_left_angle;
	int x_dest, y_dest;

	// search for the direction with maximal response from van_pt_ini
	int max_angle = max(60, ((int)(atan(max(img_size.width-van_pt_ini.x, van_pt_ini.x)/(y_bottom_warp - van_pt_ini.y))*180/CV_PI)/5+1)*5);
	cout << "max_angle: " << max_angle << endl;
	for (int i = 5; i <= max_angle; i += 5)
	{
		Mat mask_ray(img_size, CV_8UC1, Scalar(0));
		x_dest = van_pt_ini.x + (y_bottom_warp-van_pt_ini.y)*tan(-direction*i*CV_PI/180);
		y_dest = y_bottom_warp;
		if (x_dest <0 && direction == 1) // || x_dest >= img_size.width)
		{
			x_dest = 0;
			y_dest = van_pt_ini.y + (van_pt_ini.x - x_dest)/tan(direction*i*CV_PI/180);
		}
		else if (x_dest >= img_size.width && direction == -1)
		{
			x_dest = img_size.width - 1;
			y_dest = van_pt_ini.y + (van_pt_ini.x - x_dest)/tan(direction*i*CV_PI/180);
		}

		Point dest(x_dest, y_dest);
		line(mask_ray, Point(van_pt_ini), dest, Scalar(255), 10 );
		Point dist = Point(van_pt_ini) - dest;
		float length_line = sqrt(dist.x*dist.x + dist.y*dist.y);

		Mat masked_steer_resp;
		steer_resp_mag.copyTo(masked_steer_resp, mask_ray);

		Scalar sum_steer_sc = sum(masked_steer_resp);

		float sum_steer = sum_steer_sc[0]/length_line;

		if (sum_steer > max_sum_left_steer)
		{
			max_sum_left_steer = sum_steer;
			max_left_angle = i;
			mask_ray.copyTo(mask_side);
		}
	}

	// refine the found direction with finer grid
	int curr_max_angle = max_left_angle;
	for (int i = curr_max_angle-3; i <= curr_max_angle+3; i += 2)
	{
		Mat mask_ray(img_size, CV_8UC1, Scalar(0));
		x_dest = van_pt_ini.x + (y_bottom_warp-van_pt_ini.y)*tan(-direction*i*CV_PI/180);
		y_dest = y_bottom_warp;
		if (x_dest <0) // || x_dest >= img_size.width)
		{
			x_dest = 0;
			y_dest = van_pt_ini.y + (van_pt_ini.x - x_dest)/tan(direction*i*CV_PI/180);
		}
		Point dest(x_dest, y_dest);
		line(mask_ray, Point(van_pt_ini), dest, Scalar(255), 15 );
		Point dist = Point(van_pt_ini) - dest;
		float length_line = sqrt(dist.x*dist.x + dist.y*dist.y);

		Mat masked_steer_resp;
		steer_resp_mag.copyTo(masked_steer_resp, mask_ray);

		Scalar sum_steer_sc = sum(masked_steer_resp);

		float sum_steer = sum_steer_sc[0]/length_line;

		if (sum_steer > max_sum_left_steer)
		{
			max_sum_left_steer = sum_steer;
			max_left_angle = i;
			mask_ray.copyTo(mask_side);
		}
	}

	y_bottom_warp = min(y_bottom_warp, y_dest);
}

void VanPt::NMS(Mat matrix, Mat& matrix_nms)
{
	for (int i = img_size.height/2; i < y_bottom_warp; i++)
	{
		int j = 2;
		uchar* rowptr = matrix.ptr<uchar>(i);
		uchar* rowptr_nms = matrix_nms.ptr<uchar>(i);
		
		while (j < img_size.width-2)
		{
			if (rowptr[j] > rowptr[j+1] )
			{
				if (rowptr[j] >= rowptr[j-1])
				{
					rowptr_nms[j] = rowptr[j];
					// rowptr_nms[j+1] = rowptr[j+1];
					// rowptr_nms[j-1] = rowptr[j-1];
					// rowptr_nms[j+2] = rowptr[j+2];
					// rowptr_nms[j-2] = rowptr[j-2];
					
				}
			}
			else
			{
				j++;
				while(j < img_size.width-2 && rowptr[j] <= rowptr[j+1])
				{
					j++;
				}
				if (j < img_size.width-2)
				{
					rowptr_nms[j] = rowptr[j];
					// rowptr_nms[j+1] = rowptr[j+1];
					// rowptr_nms[j-1] = rowptr[j-1];
					// rowptr_nms[j+2] = rowptr[j+2];
					// rowptr_nms[j-2] = rowptr[j-2];
				}
			}
			j += 2;
		}
	}
}


void VanPt::DecideChnlThresh(Mat color_img, Mat image, Mat blur_edges) // use normalized one or ?
{
/// generate thresholds for color channels
	Mat pixhis_b, pixhis_g, pixhis_r;
	Mat pixhis_h, pixhis_l, pixhis_s;
	Mat img_hls(color_img.size(), CV_8UC3);
	cvtColor(color_img, img_hls, COLOR_BGR2HLS);
	vector<Mat> bgr_chnl, hls_chnl;
	split(color_img, bgr_chnl);
	split(img_hls, hls_chnl);
	
	
	
	#ifndef HIGH_BOT
	blur_edges.rowRange(0, image.rows*2/3) = 0;
	#else
	blur_edges.rowRange(0, image.rows/2) = 0;
	blur_edges.rowRange(image.rows*7/10, image.rows) = 0;  // for caltech data
	#endif
	
    Mat mask_trap(image.size(), CV_8UC1, Scalar(0));
    drawContours(mask_trap, warp_test_vec, -1, Scalar(255), CV_FILLED );
    
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
	localMaxima(pixhis_b, 20, 15, 0, local_max_pt[0], local_max[0], chnl_thresh[0], "b");
	localMaxima(pixhis_g, 20, 15, 0, local_max_pt[1], local_max[1], chnl_thresh[1], "g");
	localMaxima(pixhis_r, 20, 15, 0, local_max_pt[2], local_max[2], chnl_thresh[2], "r");
	localMaxima(pixhis_h, 10, 15, 0, local_max_pt[3], local_max[3], chnl_thresh[3], "h");
	localMaxima(pixhis_l, 25, 15, 0, local_max_pt[4], local_max[4], chnl_thresh[4], "l");
	localMaxima(pixhis_s, 25, 15, 0, local_max_pt[5], local_max[5], chnl_thresh[5], "s");
	*/
	localMaxima(pixhis_b, num_nonz, chnl_thresh[0], hist_size);
	localMaxima(pixhis_g, num_nonz, chnl_thresh[1], hist_size);
	localMaxima(pixhis_r, num_nonz, chnl_thresh[2], hist_size);
	localMaxima(pixhis_h, num_nonz, chnl_thresh[3], hist_size_h);
	localMaxima(pixhis_l, num_nonz, chnl_thresh[4], hist_size);
	localMaxima(pixhis_s, num_nonz, chnl_thresh[5], hist_size);
	
	cout << "finish finding peaks: " << chnl_thresh[0] << " " << chnl_thresh[1] << " " << chnl_thresh[2] << " " << chnl_thresh[3] << " " << chnl_thresh[4] << " " << chnl_thresh[5] << " " << endl;
	
	
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
  
	line(hist_image_bgr, Point(chnl_thresh[0]*bin_w, 0), Point(chnl_thresh[0]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_bgr, Point(chnl_thresh[1]*bin_w, 0), Point(chnl_thresh[1]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_bgr, Point(chnl_thresh[2]*bin_w, 0), Point(chnl_thresh[2]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
  
	line(hist_image_hls, Point(chnl_thresh[3]*bin_wh, 0), Point(chnl_thresh[3]*bin_wh, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_hls, Point(chnl_thresh[4]*bin_w, 0), Point(chnl_thresh[4]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
	line(hist_image_hls, Point(chnl_thresh[5]*bin_w, 0), Point(chnl_thresh[5]*bin_w, hist_h-1), Scalar(255, 255,255), 1, 4);
  
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
	
	color_mask = bgr_chnl[0] > chnl_thresh[0] & bgr_chnl[0] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked b", masked_edges );
	
	color_mask = bgr_chnl[1] > chnl_thresh[1] & bgr_chnl[1] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked g", masked_edges );
	
	color_mask = bgr_chnl[2] > chnl_thresh[2] & bgr_chnl[2] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked r", masked_edges );
	
	color_mask = hls_chnl[0] > 0 & hls_chnl[0] < chnl_thresh[3];
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked h", masked_edges );
	
	color_mask = hls_chnl[1] > chnl_thresh[4] & hls_chnl[1] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked l", masked_edges );
	
	color_mask = hls_chnl[2] > chnl_thresh[5] & hls_chnl[2] < 255;
	bitwise_and(image, color_mask, masked_edges);
	bitwise_and(masked_edges, height_mask, masked_edges);
	imshow("masked s", masked_edges );
	
	Mat blur_edges_bi = blur_edges > 0;
	bitwise_and(image, blur_edges_bi, masked_edges);
	imshow("masked edge", masked_edges );
	waitKey(0);
	
	#endif
}