#include "LaneImage.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

void LaneImage::__fitLaneMovingWindow(int& hist_width)
{
	///based on the warped binary output
	
	time_t t_temp1 = clock();
	// __lane_window_out_img is for visualizing the result
	Mat warped_filter_image_U;
	__warped_filter_image.convertTo(warped_filter_image_U, CV_8U, 255, 0 );
	int from_to[] = {0,0, 0,1, 0,2};
	Mat out[] = {__lane_window_out_img, __lane_window_out_img, __lane_window_out_img};
	mixChannels(&warped_filter_image_U, 1, out, 3, from_to, 3);
	
	float frow = (float)warp_row;
	
	/// find all non-zero pixels
	vector<Point> nonz_loc;
	findNonZero(warped_filter_image_U, nonz_loc);
	
	cout << "# of non-zero pixels: " << nonz_loc.size() << endl;
	if (nonz_loc.size() <= __window_min_pixel) /// safe return
	{
		cout << "This frame failed: no non-zero pixel found" << endl;
		return;
	}
	
	valarray<float> nonzx(nonz_loc.size()), nonzy(nonz_loc.size());
	for (vector<Point>::iterator it = nonz_loc.begin() ; it != nonz_loc.end(); it++)
	{
		nonzx[it - nonz_loc.begin()] = (*it).x;
		nonzy[it - nonz_loc.begin()] = (*it).y;
	}
	valarray<bool> left_lane_inds(false, nonzx.size());
	valarray<bool> right_lane_inds(false, nonzx.size());
	 
	time_t t_temp2 = clock();
	 
	/// find interested pixels
	__laneBase(hist_width);
	__ROIInds(__window_half_width, nonzx, nonzy, left_lane_inds, right_lane_inds);
	
	time_t t_temp3 = clock();
	cout << "Toarray: " << to_string(((float)(t_temp2 - t_temp1))/CLOCKS_PER_SEC) << "s. Window: ";
			cout << to_string(((float)(t_temp3 - t_temp2))/CLOCKS_PER_SEC) <<"s. Total: ";
			cout << to_string(((float)(t_temp3 - t_temp1))/CLOCKS_PER_SEC) <<"s. " << endl << endl;
	
	/// fit the interested pixels (start to loop)
	Mat lane_fit;
	Mat left_fit, right_fit; // for first time finding out which lane is closer to history
	int iteration_subsample_num = 3;
	
	srand (time(NULL)); // for sampling if oneside dominates
	bool normal = true; // for indicating whether abnormal cases are found
	for (int iteration_subsample = 0; iteration_subsample < iteration_subsample_num; iteration_subsample++)
	{
		time_t t_temp4 = clock();
		/// transform the pixels from valarray into Mat
		valarray<float> leftx(nonzx[left_lane_inds]);
		valarray<float> lefty(nonzy[left_lane_inds]);
		valarray<float> rightx(nonzx[right_lane_inds]);
		valarray<float> righty(nonzy[right_lane_inds]);
		
		cout << "# of left pixels:  " << leftx.size() << endl;
		cout << "# of right pixels: " << rightx.size() << endl;
		/// safe return
		if (leftx.size() <= __window_min_pixel || rightx.size() <= __window_min_pixel || 
			lefty.min() >= warp_row*4/5 || righty.min() >= warp_row*4/5 || lefty.max() <= warp_row/2 || righty.max() <= warp_row/2 ) 
		{
			if (iteration_subsample == 0) /// allow makeUpFilter
			{
				bool left;
				if (leftx.size() > __window_min_pixel && rightx.size() <= __window_min_pixel)
				{
					cout << "Make-up filter used for right lane. "<< endl;
					left = false;
					__makeUpFilter(left, warped_filter_image_U, nonz_loc, nonzx, nonzy, hist_width, leftx, lefty, rightx, righty);
				}
				else if (leftx.size() <= __window_min_pixel && rightx.size() > __window_min_pixel)
				{
					cout << "Make-up filter used for left lane. "<< endl;
					left = true;
					__makeUpFilter(left, warped_filter_image_U, nonz_loc, nonzx, nonzy, hist_width, leftx, lefty, rightx, righty);
				}
				if (leftx.size() <= __window_min_pixel || rightx.size() <= __window_min_pixel || 
					lefty.min() >= warp_row*4/5 || righty.min() >= warp_row*4/5 || lefty.max() <= warp_row/2 || righty.max() <= warp_row/2 ) // check the conditions again
				{
					cout << "This frame failed: no lane pixel found" << endl;
					return;
				}
				else
				{
					if (left)
					{
						__left_nolane = true;
						cout << "Left lane is not a lane. "<< endl;
					}
					else
					{
						__right_nolane = true;
						cout << "Right lane is not a lane. "<< endl;
					}
				}
			}
			else
			{
				cout << "This frame failed: no lane pixel found" << endl;
				return;
			}
		}
		
		time_t t_temp5 = clock();
		
		float real_length_left = leftx.size();
		float real_length_right = rightx.size();
		float length_left = real_length_left, length_right = real_length_right;
		
		/// avoid one side dominates
		float min_strong_pixel = 600;
		float less_side_num;
		bool left_less = false, need_samp = false;
		if (1.2 * real_length_left < real_length_right)
		{
			if (real_length_left > min_strong_pixel)
			{
				length_right = real_length_left;
				left_less = true;
				need_samp = true;
			}
		}
		else if (1.2 * real_length_right < real_length_left)
		{
			if (real_length_right > min_strong_pixel)
			{
				length_left = real_length_right;
				left_less = false;
				need_samp = true;
			}
		}
		
		cout << "# of left pixels fitted:  " << length_left << endl;
		cout << "# of right pixels fitted: " << length_right << endl;
		
		Mat X_lane(1, length_left + length_right, CV_32F);
		Mat Y_lane(4, length_left + length_right, CV_32F, Scalar_<float>(1));
		Mat W_lane(1, length_left + length_right, CV_32F);
		
		float* line_Y1 = Y_lane.ptr<float>(1);
		float* line_Y2 = Y_lane.ptr<float>(2);
		float* line_Y3 = Y_lane.ptr<float>(3); // for indicating whether it is left or right
		float* line_X = X_lane.ptr<float>();
		
		float* line_W = W_lane.ptr<float>();
		
		
		if (iteration_subsample == iteration_subsample_num - 1) // only in last iteration
		{
			if (need_samp && !left_less)
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = rand() % (int)real_length_left;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
				}
				for(int i = 0; i < real_length_left; i++)
					__lane_window_out_img.at<Vec3b>(lefty[i], leftx[i]) = Vec3b(150, 0, 0);
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = ind_lane - length_left;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
					__lane_window_out_img.at<Vec3b>(righty[i], rightx[i]) = Vec3b(0, 0, 150);
				}
			}
			else if (need_samp && left_less)
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = ind_lane;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
					__lane_window_out_img.at<Vec3b>(lefty[i], leftx[i]) = Vec3b(150, 0, 0);
				}
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = rand() % (int)real_length_right;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
				}
				for(int i = 0; i < real_length_right; i++)
					__lane_window_out_img.at<Vec3b>(righty[i], rightx[i]) = Vec3b(0, 0, 150);
			}
			else
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = ind_lane;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
					__lane_window_out_img.at<Vec3b>(lefty[i], leftx[i]) = Vec3b(150, 0, 0);
				}
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = ind_lane - length_left;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
					__lane_window_out_img.at<Vec3b>(righty[i], rightx[i]) = Vec3b(0, 0, 150);
				}
			}
		}
		else
		{
			if (need_samp && !left_less)
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = rand() % (int)real_length_left;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
				}
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = ind_lane - length_left;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
				}
			}
			else if (need_samp && left_less)
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = ind_lane;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
				}
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = rand() % (int)real_length_right;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
				}
			}
			else
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					int i = ind_lane;
					line_Y1[ind_lane] = (frow - 1 - lefty[i]);
					line_Y2[ind_lane] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
					line_Y3[ind_lane] = -1;
					line_X[ind_lane] = leftx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
					//line_Wl[i] = 1;
				}
				
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					int i = ind_lane - length_left;
					line_Y1[ind_lane] = (frow - 1 -righty[i]);
					line_Y2[ind_lane] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
					line_Y3[ind_lane] = 1; //is 1 at construction
					line_X[ind_lane] = rightx[i];
					line_W[ind_lane] = 8.0*( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
					//line_Wr[i] = 1;
				}
			}
		}
		
		time_t t_temp6 = clock();
		
		/// start robust regression
		
			left_fit = (Y_lane(Range(0, 3), Range(0, length_left)).t()).inv(DECOMP_SVD)*(X_lane.colRange(0, length_left).t());
			right_fit = (Y_lane(Range(0, 3), Range(length_left, length_left + length_right)).t()).inv(DECOMP_SVD)*(X_lane.colRange(length_left, length_left + length_right).t());
			
			Mat res_lane(X_lane.cols, 1, CV_32FC1);
			res_lane.rowRange(0, length_left) = (abs(X_lane.colRange(0, length_left) - left_fit.at<float>(2)*Y_lane(Range(2,3), Range(0, length_left)) 
				- left_fit.at<float>(1)*Y_lane(Range(1,2), Range(0, length_left)) - left_fit.at<float>(0))).t() + 0.1;
			res_lane.rowRange(length_left, length_left+length_right) = (abs(X_lane.colRange(length_left, length_left+length_right) 
				- left_fit.at<float>(2)*Y_lane(Range(2,3), Range(length_left, length_left+length_right)) - left_fit.at<float>(1)*Y_lane(Range(1,2), Range(length_left, length_left+length_right)) - left_fit.at<float>(0))).t() + 0.1;
			
			res_lane = 5/res_lane;
			
			if (__avg_hist_left_fit != Vec3f(0, 0, 0) && __avg_hist_right_fit != Vec3f(0, 0, 0))
			{
				if (iteration_subsample == 0 || normal)
				{
					Vec3f left_fit_vec = left_fit;
					Vec3f right_fit_vec = right_fit;
					__left_dist_to_hist = __getDiff(left_fit_vec, __avg_hist_left_fit);
					__right_dist_to_hist = __getDiff(right_fit_vec, __avg_hist_right_fit);
					__left_curve_dist_to_hist = __getCurveDiff(left_fit_vec, __avg_hist_left_fit);
					__right_curve_dist_to_hist = __getCurveDiff(right_fit_vec, __avg_hist_right_fit);
				}
			}
			if (__left_curve_dist_to_hist > 0.0003 && __right_curve_dist_to_hist < 0.00015)
			{
				normal = false;
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					if (line_Y1[ind_lane] > warp_row/2) // upper part
						line_W[ind_lane] = 0;
				}
			}
			else if (__left_curve_dist_to_hist < 0.00015 && __right_curve_dist_to_hist > 0.0003)
			{
				normal = false;
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					if (line_Y1[ind_lane] > warp_row/2) // upper part
						line_W[ind_lane] = 0;
				}
			}
			else
			{
				if (__left_dist_to_hist < 0 && __right_dist_to_hist > 0)// left is bifurcating (__left_dist_to_hist >  __right_dist_to_hist)
				{
					normal = false;
					for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
					{
						if (line_Y1[ind_lane] > warp_row/2) // upper part
							line_W[ind_lane] = 0;
					}
				}
				else if (__left_dist_to_hist > 0 && __right_dist_to_hist < 0)// right is bifurcating(__right_dist_to_hist >  __left_dist_to_hist)
				{
					normal = false;
					for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
					{
						if (line_Y1[ind_lane] > warp_row/2) // upper part
							line_W[ind_lane] = 0;
					}
				}
			}
			Mat w_lane(res_lane.rows, 4, CV_32F);
			w_lane.col(0) = (5/res_lane.col(0)).mul(W_lane.t());
			w_lane.col(1) = w_lane.col(0) + 0;
			w_lane.col(2) = w_lane.col(0) + 0;
			w_lane.col(3) = w_lane.col(0) + 0;
				
			lane_fit = (Y_lane.mul(w_lane.t())*Y_lane.t()).inv()*Y_lane.mul(w_lane.t())*X_lane.t();
		/*
		else
		{
			lane_fit = (Y_lane.t()).inv(DECOMP_SVD)*(X_lane.t());
		
			if (__left_dist_to_hist < 0 && __right_dist_to_hist > 0)// left is bifurcating (__left_dist_to_hist >  __right_dist_to_hist)
			{
				for (int ind_lane = 0; ind_lane < length_left; ind_lane++)
				{
					if (line_Y1[ind_lane] > warp_row/2) // upper part
						line_W[ind_lane] = 0;
				}
				/*
				res_lane.rowRange(length_left, length_left + length_right) *= 4;
				if (__left_dist_to_hist >  2*__right_dist_to_hist)	
					res_lane.rowRange(length_left, length_left + length_right) *= 4;
				*/
				/*
			}
			else if (__left_dist_to_hist > 0 && __right_dist_to_hist < 0)// right is bifurcating(__right_dist_to_hist >  __left_dist_to_hist)
			{
				for (int ind_lane = length_left; ind_lane < length_left + length_right; ind_lane++)
				{
					if (line_Y1[ind_lane] > warp_row/2) // upper part
						line_W[ind_lane] = 0;
				}
				/*
				res_lane.rowRange(0, length_left) *= 4;
				if (__right_dist_to_hist >  2* __left_dist_to_hist)
					res_lane.rowRange(0, length_left) *= 4;
				*/
				/*
			}
		}
		*/
		if (iteration_subsample == iteration_subsample_num - 1) // last iteration is different
		{
			int iteration_num = 2;
			Mat res_lane;
			for (int iteration = 0; iteration < iteration_num; iteration++)
			{
				res_lane = (abs(X_lane - lane_fit.at<float>(2)*Y_lane.row(2) - lane_fit.at<float>(1)*Y_lane.row(1) - lane_fit.at<float>(0) - lane_fit.at<float>(3)*Y_lane.row(3))).t() + 0.1;
				
				res_lane = 1/res_lane + (frow - 1 - Y_lane.row(1)).t()*0.1; //(frow - 1 - Y_left.row(1))large for downside
				Mat w_lane(res_lane.rows, 4, CV_32F);
				w_lane.col(0) = res_lane.col(0).mul(W_lane.t());
				w_lane.col(1) = w_lane.col(0) + 0;
				w_lane.col(2) = w_lane.col(0) + 0;
				w_lane.col(3) = w_lane.col(0) + 0;
				
				lane_fit = (Y_lane.mul(w_lane.t())*Y_lane.t()).inv()*Y_lane.mul(w_lane.t())*X_lane.t();
			}
			__left_fit = lane_fit.rowRange(0,3); // Mat(1,3) to Vec3f, OK???
			__left_fit(0) = __left_fit(0) - lane_fit.at<float>(3);
			__right_fit = lane_fit.rowRange(0,3);
			__right_fit(0) = __right_fit(0) + lane_fit.at<float>(3);
			__leftx = leftx;
			__lefty = lefty;
			__rightx = rightx;
			__righty = righty;
			
			/// also train a linear model
			Mat Y_lane_lin(3, Y_lane.cols, CV_32F);
			Y_lane_lin.rowRange(0,2) = Y_lane.rowRange(0, 2) + 0;
			Y_lane_lin.row(2) = Y_lane.row(3) + 0;
			Mat lane_fit_2 = (Y_lane_lin.t()).inv(DECOMP_SVD)*(X_lane.t());
			
			__left_fit_2 = lane_fit_2.rowRange(0, 2);
			__left_fit_2(0) = __left_fit_2(0) - lane_fit_2.at<float>(2);
			__right_fit_2 = lane_fit_2.rowRange(0, 2);
			__right_fit_2(0) = __right_fit_2(0) + lane_fit_2.at<float>(2);
			

			__left_fit_cr[0] = __left_fit[0]*xm_per_pix;
			__left_fit_cr[1] = __left_fit[1]*xm_per_pix/ym_per_pix;
			__left_fit_cr[2] = __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			__right_fit_cr[0] = __right_fit[0]*xm_per_pix;
			__right_fit_cr[1] = __right_fit[1]*xm_per_pix/ym_per_pix;
			__right_fit_cr[2] = __right_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			
			
			/// train decision tree
			__laneSanityCheck(hist_width);
			
			if ( (__parallel_check && __width_check) || __nframe < 5)
			{
				__train_or_not = true;
				trainmodel(warped_filter_image_U, nonzx, nonzy, left_lane_inds, right_lane_inds);
			}
			

			//cout << "real formula from trans: [" << __left_fit[0]*xm_per_pix << ", " << __left_fit[1]*xm_per_pix/ym_per_pix << ", " << __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix << "]" << endl;
			// Mat(1,3) to Vec3f, OK???	
			break;
		}
		
		int iteration_num = 5;
		
		
		for (int iteration = 0; iteration < iteration_num; iteration++)
		{
				Mat res_lane = (abs(X_lane - lane_fit.at<float>(2)*Y_lane.row(2) - lane_fit.at<float>(1)*Y_lane.row(1) - lane_fit.at<float>(0) - lane_fit.at<float>(3)*Y_lane.row(3))).t() + 0.1;
					
				Mat w_lane(res_lane.rows, 4, CV_32F);
				w_lane.col(0) = (5/res_lane.col(0)).mul(W_lane.t());
				w_lane.col(1) = w_lane.col(0) + 0;
				w_lane.col(2) = w_lane.col(0) + 0;
				w_lane.col(3) = w_lane.col(0) + 0;
				
				lane_fit = (Y_lane.mul(w_lane.t())*Y_lane.t()).inv()*Y_lane.mul(w_lane.t())*X_lane.t();
		}
		time_t t_temp7 = clock();
		
		/// renew interested pixels
		float decay_coeff = 0.5; // 0.4
		float left_fit_at[3] = {lane_fit.at<float>(0) - lane_fit.at<float>(3), lane_fit.at<float>(1), lane_fit.at<float>(2)};
		float right_fit_at[3] = {lane_fit.at<float>(0) + lane_fit.at<float>(3), lane_fit.at<float>(1), lane_fit.at<float>(2)};
		
		left_lane_inds = (nonzx > left_fit_at[2]*(frow - 1 - nonzy)*(frow - 1 - nonzy) + left_fit_at[1]*(frow - 1 - nonzy) + left_fit_at[0] - __window_half_width*decay_coeff) 
		& (nonzx < left_fit_at[2]*(frow - 1 - nonzy)*(frow - 1 - nonzy) + left_fit_at[1]*(frow - 1 - nonzy) + left_fit_at[0] + __window_half_width*decay_coeff);
		
		right_lane_inds = (nonzx > right_fit_at[2]*(frow - 1 - nonzy)*(frow - 1 - nonzy) + right_fit_at[1]*(frow - 1 - nonzy) + right_fit_at[0] - __window_half_width*decay_coeff) 
		& (nonzx < right_fit_at[2]*(frow - 1 - nonzy)*(frow - 1 - nonzy) + right_fit_at[1]*(frow - 1 - nonzy) + right_fit_at[0] + __window_half_width*decay_coeff);	
		
		time_t t_temp8 = clock();
		cout << "Newarray: " << to_string(((float)(t_temp5 - t_temp4))/CLOCKS_PER_SEC) << "s. Tomat: ";
			cout << to_string(((float)(t_temp6 - t_temp5))/CLOCKS_PER_SEC) <<"s. Regression: ";
			cout << to_string(((float)(t_temp7 - t_temp6))/CLOCKS_PER_SEC) <<"s. Renewed: ";
			cout << to_string(((float)(t_temp8 - t_temp7))/CLOCKS_PER_SEC) <<"s. Total: ";
			cout << to_string(((float)(t_temp8 - t_temp4))/CLOCKS_PER_SEC) << endl;
		cout << "TTotal: " << to_string(((float)(t_temp8 - t_temp1))/CLOCKS_PER_SEC) << "s" << endl;
	// from downside
	}
	cout << endl;
	return;
}



void LaneImage::trainmodel(Mat& warped_filter_image_U, valarray<float>& nonzx, valarray<float>& nonzy, valarray<bool>& left_lane_inds, valarray<bool>& right_lane_inds)
{
	float frow = (float)warp_row;
			/// determine the size of samples of each categories
			time_t t_temp1 = clock();
			
			valarray<bool> left_up_inds = __lefty < frow/2;
			valarray<bool> right_up_inds = __righty < frow/2;
			valarray<bool> left_down_inds = !left_up_inds;
			valarray<bool> right_down_inds = !right_up_inds;
			
			valarray<float> leftx_up(__leftx[left_up_inds]);
			valarray<float> lefty_up(__lefty[left_up_inds]);
			valarray<float> leftx_down(__leftx[left_down_inds]);
			valarray<float> lefty_down(__lefty[left_down_inds]);
			valarray<float> rightx_up(__rightx[right_up_inds]);
			valarray<float> righty_up(__righty[right_up_inds]);
			valarray<float> rightx_down(__rightx[right_down_inds]);
			valarray<float> righty_down(__righty[right_down_inds]);
			
			float length_left_up = leftx_up.size();
			float length_left_down = leftx_down.size();
			float length_right_up = rightx_up.size();
			float length_right_down = rightx_down.size();
			
			float length_left = length_left_up + length_left_down;
			float length_right = length_right_up + length_right_down;
			
			Mat lane_window_out_r(warp_row, warp_col, CV_8UC1);
			Mat lane_window_out_l(warp_row, warp_col, CV_8UC1);
			Mat out[] = { lane_window_out_r, lane_window_out_l };
			int from_to[] = { 2,0 , 0,1 };
			mixChannels( &__lane_window_out_img, 1, out, 2, from_to, 2 );

			Mat warped_filter_image_flip = (warped_filter_image_U == 0);
			vector<Point> zero_loc;
			findNonZero(warped_filter_image_flip, zero_loc);
			float length_flip = zero_loc.size();
			
			Mat warped_filter_peak_ = __warped_filter_image > 0.9; // only the most possible points
			Mat warped_filter_peak_r = warped_filter_peak_ & (lane_window_out_r == 150);
			Mat warped_filter_peak_l = warped_filter_peak_ & (lane_window_out_l == 150);
			
			Mat up_mask(warped_filter_peak_.size(), CV_8UC1, Scalar(0));
			up_mask.rowRange(0, warp_row/2) = 255;
			Mat warped_filter_peak_l_up = warped_filter_peak_l & up_mask;
			Mat warped_filter_peak_l_down = warped_filter_peak_l & (~up_mask);
			Mat warped_filter_peak_r_up = warped_filter_peak_r & up_mask;
			Mat warped_filter_peak_r_down = warped_filter_peak_r & (~up_mask);
			
			vector<Point> peak_loc_r_up, peak_loc_r_down, peak_loc_l_up, peak_loc_l_down;
			findNonZero(warped_filter_peak_r_up, peak_loc_r_up);
			findNonZero(warped_filter_peak_r_down, peak_loc_r_down);
			findNonZero(warped_filter_peak_l_up, peak_loc_l_up);
			findNonZero(warped_filter_peak_l_down, peak_loc_l_down);
			
			float length_peak_r_up = peak_loc_r_up.size();
			float length_peak_r_down = peak_loc_r_down.size();
			float length_peak_l_up = peak_loc_l_up.size();
			float length_peak_l_down = peak_loc_l_down.size();
			
			valarray<bool> non_lane_inds = !(left_lane_inds || right_lane_inds);
			valarray<float> notlane_x_pre = nonzx[ non_lane_inds ];
			valarray<float> notlane_y_pre = nonzy[ non_lane_inds ];
			
			valarray<bool> close_left_inds = (notlane_x_pre > __left_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __left_fit[1]*(frow - 1 - notlane_y_pre) + __left_fit[0] - 2*__window_half_width) 
		& (notlane_x_pre < __left_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __left_fit[1]*(frow - 1 - notlane_y_pre) + __left_fit[0] + 2*__window_half_width);
			valarray<bool> close_right_inds = (notlane_x_pre > __right_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __right_fit[1]*(frow - 1 - notlane_y_pre) + __right_fit[0] - 2*__window_half_width) 
		& (notlane_x_pre < __right_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __right_fit[1]*(frow - 1 - notlane_y_pre) + __right_fit[0] + 2*__window_half_width);
			valarray<bool> close_lane_inds = close_left_inds || close_right_inds;
			
			valarray<float> notlane_x(notlane_x_pre[close_lane_inds]);
			valarray<float> notlane_y(notlane_y_pre[close_lane_inds]);
			
			float length_notlane = notlane_x.size();
			
			cout << "length_peak_l_up: " << length_peak_l_up << ", l_down: " << length_peak_l_down << ", r_up: " << length_peak_r_up << ", r_down: " << length_peak_r_down << ", length_notlane: " << length_notlane << ", length_flip: " << length_flip << endl;
			/*
			int n_samp = warp_col * warp_row / 400; // 2000 for 1280*720, 500 for 400*500
			int n_samp_pos = n_samp / 4;
			int n_samp_neg = n_samp - n_samp_pos;
			int n_samp_pos_l = length_left/ (length_left + length_right) *n_samp_pos;
			int n_samp_pos_r = n_samp_pos - n_samp_pos_l; 
			*/
			
			int n_samp = __BGR_resp.rows / 5; // 500 for 1280*720, 100 for 400*500
			int n_samp_pos = n_samp / 4;
			int n_samp_neg = n_samp - n_samp_pos;
			int n_samp_pos_l_up,n_samp_pos_l_down, n_samp_pos_r_up, n_samp_pos_r_down; 
			if (__left_nolane)
			{
				n_samp_pos_l_up = 0;
				n_samp_pos_l_down = 0;
				int n_samp_pos_r = n_samp_pos;
				if (length_right_up == 0)
				{
					n_samp_pos_r_up = 0;
					n_samp_pos_r_down = n_samp_pos_r;
				}
				else if (length_right_down == 0)
				{
					n_samp_pos_r_down = 0;
					n_samp_pos_r_up = n_samp_pos_r;
				}
				else if (length_right_up < length_right_down)
				{
					n_samp_pos_r_up = min( length_right_up, max((float)0.2, length_right_up / length_right)*n_samp_pos_r);
					n_samp_pos_r_down = n_samp_pos_r - n_samp_pos_r_up;
				}
				else
				{
					n_samp_pos_r_down = min( length_right_down, max((float)0.2, length_right_down / length_right)*n_samp_pos_r);
					n_samp_pos_r_up = n_samp_pos_r - n_samp_pos_r_down;
				}
				
			}
			else if (__right_nolane)
			{
				n_samp_pos_r_up = 0;
				n_samp_pos_r_down = 0;
				int n_samp_pos_l = n_samp_pos;
				if (length_left_up == 0)
				{
					n_samp_pos_l_up = 0;
					n_samp_pos_l_down = n_samp_pos_l;
				}
				else if (length_left_down == 0)
				{
					n_samp_pos_l_down = 0;
					n_samp_pos_l_up = n_samp_pos_l;
				}
				else if (length_left_up < length_left_down)
				{
					n_samp_pos_l_up = min( length_left_up, max((float)0.2, length_left_up / length_left)*n_samp_pos_l);
					n_samp_pos_l_down = n_samp_pos_l - n_samp_pos_l_up;
				}
				else
				{
					n_samp_pos_l_down = min( length_left_down, max((float)0.2, length_left_down / length_left)*n_samp_pos_l);
					n_samp_pos_l_up = n_samp_pos_l - n_samp_pos_l_down;
				}
			}
			else
			{
				int n_samp_pos_l = min( (float)0.8, max(length_left/ (length_left + length_right), (float) 0.2) ) *n_samp_pos;
				int n_samp_pos_r = n_samp_pos - n_samp_pos_l; 
				/// left
				if (length_left_up == 0)
				{
					n_samp_pos_l_up = 0;
					n_samp_pos_l_down = n_samp_pos_l;
				}
				else if (length_left_down == 0)
				{
					n_samp_pos_l_down = 0;
					n_samp_pos_l_up = n_samp_pos_l;
				}
				else if (length_left_up < length_left_down)
				{
					n_samp_pos_l_up = min( length_left_up, max((float)0.2, length_left_up / length_left)*n_samp_pos_l);
					n_samp_pos_l_down = n_samp_pos_l - n_samp_pos_l_up;
				}
				else
				{
					n_samp_pos_l_down = min( length_left_down, max((float)0.2, length_left_down / length_left)*n_samp_pos_l);
					n_samp_pos_l_up = n_samp_pos_l - n_samp_pos_l_down;
				}
				/// right
				if (length_right_up == 0)
				{
					n_samp_pos_r_up = 0;
					n_samp_pos_r_down = n_samp_pos_r;
				}
				else if (length_right_down == 0)
				{
					n_samp_pos_r_down = 0;
					n_samp_pos_r_up = n_samp_pos_r;
				}
				else if (length_right_up < length_right_down)
				{
					n_samp_pos_r_up = min( length_right_up, max((float)0.2, length_right_up / length_right)*n_samp_pos_r);
					n_samp_pos_r_down = n_samp_pos_r - n_samp_pos_r_up;
				}
				else
				{
					n_samp_pos_r_down = min( length_right_down, max((float)0.2, length_right_down / length_right)*n_samp_pos_r);
					n_samp_pos_r_up = n_samp_pos_r - n_samp_pos_r_down;
				}
			}
				
			int num_left_up, num_left_down, num_right_up, num_right_down, num_notlane, num_flip, num_lane;
			int num_left, num_right;
			num_left_up = n_samp_pos_l_up;
			num_left_down = n_samp_pos_l_down;
			num_left = num_left_up + num_left_down;
			
			num_right_up = n_samp_pos_r_up;
			num_right_down = n_samp_pos_r_down;
			num_right = num_right_up + num_right_down;
			
			num_notlane = (int)(min(  length_notlane,  max((float)0.05, length_notlane/ (length_notlane + length_flip))* n_samp_neg  ));
			num_lane = num_left_up + num_left_down + num_right_up + num_right_down;
			num_flip = n_samp - num_lane - num_notlane ; // need change, how to handle extreme cases when sample points are few
			
			cout << "n_sample: " << num_left_up <<" " << num_left_down <<" " << num_right_up <<" " << num_right_down <<" " << num_lane << " " << num_notlane <<" " << num_flip << endl;
			
			
			time_t t_temp2;
			if ( __last_left_fit == Vec3f(0, 0, 0) || __last_right_fit == Vec3f(0, 0, 0) ) // __nframe == 0
			{
				/*
				int mode_for_pos_samp; // for indicating whether peak samples are enough
				if ( (length_peak_l > 5 * n_samp_pos_l || __left_nolane) && (length_peak_r > 5 * n_samp_pos_r || __right_nolane) )
					mode_for_pos_samp = 1;
				else
					mode_for_pos_samp = 0;
				*/
				bool use_peak_l_up = length_peak_l_up > 5*n_samp_pos_l_up;
				bool use_peak_l_down = length_peak_l_down > 5*n_samp_pos_l_down;
				bool use_peak_r_up = length_peak_r_up > 5*n_samp_pos_r_up;
				bool use_peak_r_down = length_peak_r_down > 5*n_samp_pos_r_down;
				cout << "mode_for_pos_samp: " << use_peak_l_up << " " << use_peak_l_down << " " << use_peak_r_up << " " << use_peak_r_down << endl;
				
				
				t_temp2 = clock();
				
				/// sort the pixels and start sampling
				
				/*Mat warp_reshape = __warped_raw_image.reshape(1, __row*__col);
				warp_reshape.convertTo(warp_reshape, CV_32FC1);
				
				Mat warp_HLS;
				cvtColor(__warped_raw_image, warp_HLS, COLOR_BGR2HLS);
				Mat warp_reshape_HLS = warp_HLS.reshape(1, __row*__col);
				warp_reshape_HLS.convertTo(warp_reshape_HLS, CV_32FC1);
				*/
				
				srand (time(NULL));
				/*
				int ind;
				for (int i = 0; i < num_left; i++)
				{
					ind = rand() % (int)length_left;
					Vec3b temp = __warped_raw_image.at<Vec3b>(lefty[ind], leftx[ind]);
					BGR_sample.at<float>(i, 0) = temp[0];
					BGR_sample.at<float>(i, 1) = temp[1];
					BGR_sample.at<float>(i, 2) = temp[2];
					cout << __lane_window_out_img.at<Vec3b>(lefty[ind], leftx[ind]) << endl;
					cout << __warped_raw_image.at<Vec3b>(lefty[ind], leftx[ind]) << endl;
					cout << warp_reshape.row(lefty[ind]*__col + leftx[ind]) << endl << endl;
					
					//cout << ind << endl;
				}
				*/
				// the sequence of reshape is row by row
				
				int ind;
				for (int j = 0; j < 5; j++)
				{
					#ifdef DTREE
					__BGR_resp.rowRange(j * n_samp, j * n_samp + num_lane) = 1;
					__BGR_resp.rowRange(j * n_samp + num_lane, (j+1) * n_samp) = 0; // end is exclusive
					__HLS_resp.rowRange(j * n_samp, j * n_samp + num_lane) = 1;
					__HLS_resp.rowRange(j * n_samp + num_lane, (j+1) * n_samp) = 0; // end is exclusive
					#endif
					#ifdef LREGG
					__BGR_resp.rowRange(j * n_samp, j * n_samp + num_left) = 1;
					__BGR_resp.rowRange(j * n_samp + num_left, j * n_samp + num_lane) = 2; // end is exclusive
					__BGR_resp.rowRange(j * n_samp + num_lane, (j+1) * n_samp) = 0;
					__HLS_resp.rowRange(j * n_samp, j * n_samp + num_left) = 1;
					__HLS_resp.rowRange(j * n_samp + num_left, j * n_samp + num_lane) = 2; // end is exclusive
					__HLS_resp.rowRange(j * n_samp + num_lane, (j+1) * n_samp) = 0;
					#endif
					/// left up
					if (use_peak_l_up)
					{
						if (length_peak_l_up != 0 && num_left_up != 0 )
							for (int i = 0; i < num_left_up; i++)
							{
								ind = rand() % (int)length_peak_l_up;
								__BGR_sample.row(i + j * n_samp) = __warped_reshape.row(peak_loc_l_up[ind].y*warp_col + peak_loc_l_up[ind].x) + 0;
								__HLS_sample.row(i + j * n_samp) = __warped_reshape_HLS.row(peak_loc_l_up[ind].y*warp_col + peak_loc_l_up[ind].x) + 0;
								__lane_window_out_img.at<Vec3b>( peak_loc_l_up[ind] ) *= 2;
							}
					}
					else
					{
						if (length_left_up != 0 && num_left_up != 0 )
							for (int i = 0; i < num_left_up; i++)
							{
								ind = rand() % (int)length_left_up;
								__BGR_sample.row(i + j * n_samp) = __warped_reshape.row(lefty_up[ind]*warp_col + leftx_up[ind]) + 0;
								__HLS_sample.row(i + j * n_samp) = __warped_reshape_HLS.row(lefty_up[ind]*warp_col + leftx_up[ind]) + 0;
								__lane_window_out_img.at<Vec3b>(lefty_up[ind], leftx_up[ind]) *= 2;
							}
					}
					///left down
					if (use_peak_l_down)
					{
						if (length_peak_l_down != 0 && num_left_down != 0 )
							for (int i = 0; i < num_left_down; i++)
							{
								ind = rand() % (int)length_peak_l_down;
								__BGR_sample.row(i + j * n_samp + num_left_up) = __warped_reshape.row(peak_loc_l_down[ind].y*warp_col + peak_loc_l_down[ind].x) + 0;
								__HLS_sample.row(i + j * n_samp + num_left_up) = __warped_reshape_HLS.row(peak_loc_l_down[ind].y*warp_col + peak_loc_l_down[ind].x) + 0;
								__lane_window_out_img.at<Vec3b>( peak_loc_l_down[ind] ) *= 2;
							}
					}
					else
					{
						if (length_left_down != 0 && num_left_down != 0 )
							for (int i = 0; i < num_left_down; i++)
							{
								ind = rand() % (int)length_left_down;
								__BGR_sample.row(i + j * n_samp + num_left_up) = __warped_reshape.row(lefty_down[ind]*warp_col + leftx_down[ind]) + 0;
								__HLS_sample.row(i + j * n_samp + num_left_up) = __warped_reshape_HLS.row(lefty_down[ind]*warp_col + leftx_down[ind]) + 0;
								__lane_window_out_img.at<Vec3b>(lefty_down[ind], leftx_down[ind]) *= 2;
							}
					}
					///right up
					if (use_peak_r_up)
					{
						if (length_peak_r_up != 0 && num_right_up != 0 )
							for (int i = 0; i < num_right_up; i++)
							{
								ind = rand() % (int)length_peak_r_up;
								__BGR_sample.row(i + j * n_samp + num_left) = __warped_reshape.row(peak_loc_r_up[ind].y*warp_col + peak_loc_r_up[ind].x) + 0;
								__HLS_sample.row(i + j * n_samp + num_left) = __warped_reshape_HLS.row(peak_loc_r_up[ind].y*warp_col + peak_loc_r_up[ind].x) + 0;
								__lane_window_out_img.at<Vec3b>( peak_loc_r_up[ind] ) *= 2;
							}
					}
					else
					{
						if (length_right_up != 0 && num_right_up != 0 )
							for (int i = 0; i < num_right_up; i++)
							{
								ind = rand() % (int)length_right_up;
								__BGR_sample.row(i + j * n_samp + num_left) = __warped_reshape.row(righty_up[ind]*warp_col + rightx_up[ind]) + 0;
								__HLS_sample.row(i + j * n_samp + num_left) = __warped_reshape_HLS.row(righty_up[ind]*warp_col + rightx_up[ind]) + 0;
								__lane_window_out_img.at<Vec3b>(righty_up[ind], rightx_up[ind]) *= 2;
							}
					}
					///right down
					if (use_peak_r_down)
					{
						if (length_peak_r_down != 0 && num_right_down != 0 )
							for (int i = 0; i < num_right_down; i++)
							{
								ind = rand() % (int)length_peak_r_down;
								__BGR_sample.row(i + j * n_samp + num_left + num_right_up) = __warped_reshape.row(peak_loc_r_down[ind].y*warp_col + peak_loc_r_down[ind].x) + 0;
								__HLS_sample.row(i + j * n_samp + num_left + num_right_up) = __warped_reshape_HLS.row(peak_loc_r_down[ind].y*warp_col + peak_loc_r_down[ind].x) + 0;
								__lane_window_out_img.at<Vec3b>( peak_loc_r_down[ind] ) *= 2;
							}
					}
					else
					{
						if (length_right_down != 0 && num_right_down != 0 )
							for (int i = 0; i < num_right_down; i++)
							{
								ind = rand() % (int)length_right_down;
								__BGR_sample.row(i + j * n_samp + num_left + num_right_up) = __warped_reshape.row(righty_down[ind]*warp_col + rightx_down[ind]) + 0;
								__HLS_sample.row(i + j * n_samp + num_left + num_right_up) = __warped_reshape_HLS.row(righty_down[ind]*warp_col + rightx_down[ind]) + 0;
								__lane_window_out_img.at<Vec3b>(righty_down[ind], rightx_down[ind]) *= 2;
							}
					}
					/// not lane
					if (length_notlane != 0 && num_notlane != 0)
						for (int i = 0; i < num_notlane; i++)
						{
							ind = rand() % (int)length_notlane;
							__BGR_sample.row(i + num_lane + j * n_samp) = __warped_reshape.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
							__HLS_sample.row(i + num_lane + j * n_samp) = __warped_reshape_HLS.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(notlane_y[ind], notlane_x[ind]) = Vec3b(0, 255, 0);
						}
					/// zero
					for (int i = 0; i < num_flip; i++)
					{
						ind = rand() % (int)length_flip;
						__BGR_sample.row(i + num_notlane +  num_lane + j * n_samp) = __warped_reshape.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
						__HLS_sample.row(i + num_notlane +  num_lane + j * n_samp) = __warped_reshape_HLS.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
						__lane_window_out_img.at<Vec3b>( zero_loc[ind] ) = Vec3b(0, 150, 0);
					}
				}
			}
			else
			{	
				/*			
				int mode_for_pos_samp; // for indicating whether peak samples are enough
				if ( (length_peak_l > n_samp_pos_l || __left_nolane) && (length_peak_r > n_samp_pos_r || __right_nolane) )
					mode_for_pos_samp = 1;
				else
					mode_for_pos_samp = 0;
				cout << "mode_for_pos_samp: " << mode_for_pos_samp << endl;		
				*/
				
				bool use_peak_l_up = length_peak_l_up > n_samp_pos_l_up;
				bool use_peak_l_down = length_peak_l_down > n_samp_pos_l_down;
				bool use_peak_r_up = length_peak_r_up > n_samp_pos_r_up;
				bool use_peak_r_down = length_peak_r_down > n_samp_pos_r_down;
				cout << "mode_for_pos_samp: " << use_peak_l_up << " " << use_peak_l_down << " " << use_peak_r_up << " " << use_peak_r_down << endl;	
			
				t_temp2 = clock();
			
				/// sort the pixels and start sampling
				
				srand (time(NULL));
				// the sequence of reshape is row by row
				
				#ifdef DTREE
				__BGR_resp.rowRange(__samp_cyc * n_samp, __samp_cyc * n_samp + num_lane) = 1;
				__BGR_resp.rowRange(__samp_cyc * n_samp + num_lane, (__samp_cyc+1) * n_samp) = 0; // end is exclusive
				__HLS_resp.rowRange(__samp_cyc * n_samp, __samp_cyc * n_samp + num_lane) = 1;
				__HLS_resp.rowRange(__samp_cyc * n_samp + num_lane, (__samp_cyc+1) * n_samp) = 0; // end is exclusive
				#endif
				#ifdef LREGG
				__BGR_resp.rowRange(__samp_cyc * n_samp, __samp_cyc * n_samp + num_left) = 1;
				__BGR_resp.rowRange(__samp_cyc * n_samp + num_left, __samp_cyc * n_samp + num_lane) = 2; // end is exclusive
				__BGR_resp.rowRange(__samp_cyc * n_samp + num_lane, (__samp_cyc+1) * n_samp) = 0;
				__HLS_resp.rowRange(__samp_cyc * n_samp, __samp_cyc * n_samp + num_left) = 1;
				__HLS_resp.rowRange(__samp_cyc * n_samp + num_left, __samp_cyc * n_samp + num_lane) = 2; // end is exclusive
				__HLS_resp.rowRange(__samp_cyc * n_samp + num_lane, (__samp_cyc+1) * n_samp) = 0;
				#endif
					
				int ind;
				/// left up
				if (use_peak_l_up)
				{
					if (length_peak_l_up != 0 && num_left_up != 0 )
						for (int i = 0; i < num_left_up; i++)
						{
							ind = rand() % (int)length_peak_l_up;
							__BGR_sample.row(i + __samp_cyc * n_samp) = __warped_reshape.row(peak_loc_l_up[ind].y*warp_col + peak_loc_l_up[ind].x) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp) = __warped_reshape_HLS.row(peak_loc_l_up[ind].y*warp_col + peak_loc_l_up[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_l_up[ind] ) *= 2;
						}
				}
				else
				{
					if (length_left_up != 0 && num_left_up != 0 )
						for (int i = 0; i < num_left_up; i++)
						{
							ind = rand() % (int)length_left_up;
							__BGR_sample.row(i + __samp_cyc * n_samp) = __warped_reshape.row(lefty_up[ind]*warp_col + leftx_up[ind]) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp) = __warped_reshape_HLS.row(lefty_up[ind]*warp_col + leftx_up[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(lefty_up[ind], leftx_up[ind]) *= 2;
						}
				}
				///left down
				if (use_peak_l_down)
				{
					if (length_peak_l_down != 0 && num_left_down != 0 )
						for (int i = 0; i < num_left_down; i++)
						{
							ind = rand() % (int)length_peak_l_down;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left_up) = __warped_reshape.row(peak_loc_l_down[ind].y*warp_col + peak_loc_l_down[ind].x) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left_up) = __warped_reshape_HLS.row(peak_loc_l_down[ind].y*warp_col + peak_loc_l_down[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_l_down[ind] ) *= 2;
						}
				}
				else
				{
					if (length_left_down != 0 && num_left_down != 0 )
						for (int i = 0; i < num_left_down; i++)
						{
							ind = rand() % (int)length_left_down;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left_up) = __warped_reshape.row(lefty_down[ind]*warp_col + leftx_down[ind]) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left_up) = __warped_reshape_HLS.row(lefty_down[ind]*warp_col + leftx_down[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(lefty_down[ind], leftx_down[ind]) *= 2;
						}
				}
				///right up
				if (use_peak_r_up)
				{
					if (length_peak_r_up != 0 && num_right_up != 0 )
						for (int i = 0; i < num_right_up; i++)
						{
							ind = rand() % (int)length_peak_r_up;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left) = __warped_reshape.row(peak_loc_r_up[ind].y*warp_col + peak_loc_r_up[ind].x) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left) = __warped_reshape_HLS.row(peak_loc_r_up[ind].y*warp_col + peak_loc_r_up[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_r_up[ind] ) *= 2;
						}
				}
				else
				{
					if (length_right_up != 0 && num_right_up != 0 )
						for (int i = 0; i < num_right_up; i++)
						{
							ind = rand() % (int)length_right_up;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left) = __warped_reshape.row(righty_up[ind]*warp_col + rightx_up[ind]) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left) = __warped_reshape_HLS.row(righty_up[ind]*warp_col + rightx_up[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(righty_up[ind], rightx_up[ind]) *= 2;
						}
				}
				///right down
				if (use_peak_r_down)
				{
					if (length_peak_r_down != 0 && num_right_down != 0 )
						for (int i = 0; i < num_right_down; i++)
						{
							ind = rand() % (int)length_peak_r_down;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left + num_right_up) = __warped_reshape.row(peak_loc_r_down[ind].y*warp_col + peak_loc_r_down[ind].x) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left + num_right_up) = __warped_reshape_HLS.row(peak_loc_r_down[ind].y*warp_col + peak_loc_r_down[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_r_down[ind] ) *= 2;
						}
				}
				else
				{
					if (length_right_down != 0 && num_right_down != 0 )
						for (int i = 0; i < num_right_down; i++)
						{
							ind = rand() % (int)length_right_down;
							__BGR_sample.row(i + __samp_cyc * n_samp + num_left + num_right_up) = __warped_reshape.row(righty_down[ind]*warp_col + rightx_down[ind]) + 0;
							__HLS_sample.row(i + __samp_cyc * n_samp + num_left + num_right_up) = __warped_reshape_HLS.row(righty_down[ind]*warp_col + rightx_down[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(righty_down[ind], rightx_down[ind]) *= 2;
						}
				}
				
				if (length_notlane != 0 && num_notlane!= 0)
					for (int i = 0; i < num_notlane; i++)
					{
						ind = rand() % (int)length_notlane;
						__BGR_sample.row(i + n_samp_pos + __samp_cyc * n_samp) = __warped_reshape.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
						__HLS_sample.row(i + n_samp_pos + __samp_cyc * n_samp) = __warped_reshape_HLS.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
						__lane_window_out_img.at<Vec3b>(notlane_y[ind], notlane_x[ind]) = Vec3b(0, 255, 0);
					}
				
				for (int i = 0; i < num_flip; i++)
				{
					ind = rand() % (int)length_flip;
					__BGR_sample.row(i + num_notlane +  n_samp_pos + __samp_cyc * n_samp) = __warped_reshape.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
					__HLS_sample.row(i + num_notlane +  n_samp_pos + __samp_cyc * n_samp) = __warped_reshape_HLS.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
					__lane_window_out_img.at<Vec3b>( zero_loc[ind] ) = Vec3b(0, 150, 0);
				}
				
			}
			time_t t_temp3 = clock();
			
			/// train decision trees
			
			//imshow("marked binary", __lane_window_out_img);
			
			#ifdef DTREE
			trainTree(__BGR_tree, __BGR_sample, __BGR_resp, __warped_reshape , warp_row , "BGR");
			trainTree(__HLS_tree, __HLS_sample, __HLS_resp, __warped_reshape_HLS , warp_row , "HLS");
			#endif
			#ifdef LREGG
			trainRegg(__BGR_regg, __BGR_sample, __BGR_resp, __warped_reshape , warp_row , "BGR");
			trainRegg(__HLS_regg, __HLS_sample, __HLS_resp, __warped_reshape_HLS , warp_row , "HLS");
			#endif
			
			time_t t_temp4 = clock();
			
			cout << "Arranged: " << to_string(((float)(t_temp2 - t_temp1))/CLOCKS_PER_SEC) << "s. Sampled: ";
			cout << to_string(((float)(t_temp3 - t_temp2))/CLOCKS_PER_SEC) << "s. Trained: " << to_string(((float)(t_temp4 - t_temp3))/CLOCKS_PER_SEC) <<"s. Total: ";
			cout << to_string(((float)(t_temp4 - t_temp1))/CLOCKS_PER_SEC) << endl;
			
			cout << "left lane: " << length_left_up << " " << length_left_down << ", right lane: " << length_right_up << " " << length_right_down << ", nonlane: " << length_notlane << ", zeroloc: " << length_flip << endl;
			cout << "total: " << length_notlane + length_flip + length_left+length_right << ". real: " << warp_row*warp_col << endl;
			
			return;
}
#ifdef DTREE
void trainTree(Ptr<ml::DTrees> tree, Mat& sample, Mat& resp, Mat& testdata, size_t orignl_row, string field)
{
	//tree = ml::DTrees::create();
			tree->setMaxDepth(4);
			tree->setMinSampleCount(20); // 20 for 400*500, 80 for 1280*720
			tree->setCVFolds(0);
			tree->setMaxCategories(2);
			tree->setUse1SERule(true);
			tree->setTruncatePrunedTree(true);
			tree->setUseSurrogates (false);
			tree->train(sample, ml::ROW_SAMPLE , resp);
			//cout << "ok. " << endl;
			/*
			vector<ml::DTrees::Split> splits = tree->getSplits();
			vector<ml::DTrees::Node> nodes = tree->getNodes();
			for (int i = 0; i < splits.size(); i++ )
			{
				cout << "idx" << i << ", varidx: " << splits[i].varIdx << ", inversed: " << splits[i].inversed << ", threshold: " << splits[i].c << ", next: " << splits[i].next << ", quality: " << splits[i].quality << endl;
			}
			for (int i = 0; i < nodes.size(); i++ )
			{
				cout << "idx" << i <<  ", value: " << nodes[i].value << ", classIdx: " << nodes[i].classIdx << ", parent: " << nodes[i].parent << ", left: " << nodes[i].left << ", right: " << nodes[i].right << ", defaultDir: " << nodes[i].defaultDir << ", split: " << nodes[i].split << endl;
			}
			*/
			
			#ifndef NDEBUG_TR
			Mat warp_response;
			tree->predict(testdata, warp_response);
			cout << "Number of nonzero response: " << countNonZero(warp_response) << endl;
			cout << "Type of response: " << warp_response.type() << endl;
			//cout << "predict: " << warp_response.rowRange(0, 500) << endl;
			//getchar();
			warp_response = warp_response.reshape(1, orignl_row);
			string window_name = "warp_response_" + field;
			imshow(window_name, warp_response);
			waitKey(0);
			#endif
			return;
}
#endif
#ifdef LREGG
void trainRegg(Ptr<ml::LogisticRegression> regg, Mat& sample, Mat& resp, Mat& testdata, size_t orignl_row, string field)
{
	int n_sample = sample.rows;
	regg->setLearningRate(0.01);
    regg->setIterations(50);
    regg->setRegularization(ml::LogisticRegression::REG_DISABLE);
    regg->setTrainMethod(ml::LogisticRegression::MINI_BATCH);
    regg->setMiniBatchSize(max(n_sample*2/50, 1));
    
    Ptr<ml::TrainData> train_data = ml::TrainData::create(sample, ml::ROW_SAMPLE, resp);
    regg->train(train_data);
    //regg->train(sample, ml::ROW_SAMPLE , resp);
    //regg->setTermCriteria( TermCriteria(EPS, 1, 0.01) );
    
			#ifndef NDEBUG_TR
			Mat warp_response;
			regg->predict(testdata, warp_response); // , ml::StatModel::RAW_OUTPUT
			cout << "Number of nonzero response: " << countNonZero(warp_response) << endl;
			cout << "Type of response: " << warp_response.type() << endl;
			warp_response.convertTo(warp_response, CV_32FC1);
			//cout << "predict: " << warp_response.rowRange(0, 500) << endl;
			//getchar();
			warp_response = warp_response.reshape(1, orignl_row);
			string window_name = "warp_response_" + field;
			imshow(window_name, warp_response);
			waitKey(0);
			#endif
			
}
#endif