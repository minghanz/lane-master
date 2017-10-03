void localMaxima(const Mat& src, const int min_peak_dist, const float min_height_diff, const float min_height, vector<int>& max_loc, vector<float>& max_val, int& min_loc, string channel)
{
	int ker_size = min_peak_dist;
	Point cur_max_pt;
	Point cur_min_pt;
	double cur_max;
	double cur_min;
	
	max_loc.clear();
	max_val.clear();
	/// find local maxima according to rule
	for (int loc = ker_size; loc < src.rows - ker_size; loc++)
	{
		Mat part = src.rowRange(loc-ker_size, loc+ker_size);
		minMaxLoc(part, &cur_min, &cur_max, NULL, &cur_max_pt);
		if (cur_max_pt.y == ker_size && cur_max - cur_min >= min_height_diff && cur_max >= min_height )
		{
			max_loc.push_back(loc);
			max_val.push_back(cur_max);
		}
	}
	
	int num_max = max_loc.size();
	if (num_max <= 0)
	{
		cout << "No peaks found for channel: " << channel << ". Use maximal value instead: ";
		minMaxLoc(src, &cur_min, &cur_max, NULL, &cur_max_pt);
		max_loc.push_back(cur_max_pt.y);
		max_val.push_back(cur_max);
		cout << "[" << max_loc[0] << ", " << max_val[0] << "] " << endl;
	}
	else
	{
		cout << num_max << "peaks found for channel " << channel << ": ";
		for (int i = 0; i < num_max; i++)
		{
			cout << "[" << max_loc[i] << ", " << max_val[i] << "] ";
		}
		cout << endl;
	}
	
	/// find desided threshold(local minima) according to rule
	if (channel != "h")
	{
		if (num_max >= 2)
		{
		Mat part = src.rowRange(max_loc[num_max-2], max_loc.back());
		minMaxLoc(part, &cur_min, NULL, &cur_min_pt, NULL);
		min_loc = max_loc[num_max-2] + cur_min_pt.y;
		}
		else
		{
			int peak_loc = max_loc.back();
			float peak_val = max_val.back();
			for (int i = 0; i < src.rows - peak_loc; i++)
			{
				if (src.at<float>(peak_loc+i) < peak_val/3)
				{
					min_loc = peak_loc + i;
					break;
				}
			}
		}
		cout << "min loc for channel " << channel <<": "<< min_loc << endl;
	}
	else
	{
		if (num_max >= 2)
		{
			Mat part = src.rowRange(max_loc[0], max_loc[1]);
			minMaxLoc(part, &cur_min, NULL, &cur_min_pt, NULL);
			min_loc = max_loc[0] + cur_min_pt.y;
		}
		else
		{
			int peak_loc = max_loc.back();
			float peak_val = max_val.back();
			for (int i = 0; i < src.rows - peak_loc; i++)
			{
				if (src.at<float>(peak_loc+i) < peak_val/3)
				{
					min_loc = peak_loc + i;
					break;
				}
			}
			cout << "not enough peaks found for channel" << channel << endl;
		}
		cout << "min loc for channel " << channel <<": "<< min_loc << endl;	
	}
	
	return;
}

void LaneImage::__fitLaneMovingWindow()
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
	float window_height = warp_row/__window_number;
	Mat nonz_loc;
	findNonZero(warped_filter_image_U, nonz_loc);
	nonz_loc = nonz_loc.reshape(1);
	
	Mat nonzx = nonz_loc.col(0);
	Mat nonzy = nonz_loc.col(1); // only create a header
	
	Mat_<int> nonzx_ = (Mat_<int>) nonzx;
	Mat_<int> nonzy_ = (Mat_<int>) nonzy;
	
	int ite = nonz_loc.rows;
	valarray<float> nonzx_vala(ite), nonzy_vala(ite); 
	valarray<bool> left_lane_inds_vala(ite), right_lane_inds_vala(ite); 
	
	Mat good_left_inds, good_right_inds;
	Mat left_lane_inds(nonz_loc.rows, 1, CV_8UC1, Scalar(0));
	Mat right_lane_inds(nonz_loc.rows, 1, CV_8UC1, Scalar(0));
	 
	time_t t_temp2 = clock();
	
	/// find interested pixels
	if (__left_fit == Vec3f(0, 0, 0))
	{
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
		
		/// extract peaks in histogram
		int midpoint = warp_col/2;
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
		minMaxIdx(histogram.colRange(0, midpoint), NULL, NULL, NULL, leftx_base_p);
		minMaxIdx(histogram.colRange(midpoint, warp_col), NULL, NULL, NULL, rightx_base_p);
		int leftx_base = leftx_base_p[1];
		int rightx_base = rightx_base_p[1] + midpoint;
		
		
		/// find pixels in windows from bottom up
		float leftx_cur = leftx_base;
		float rightx_cur = rightx_base;

		for (int i = 0; i < __window_number; i++)
		{
			float win_y_low = warp_row - (i+1)*window_height;
			float win_y_high = warp_row - i*window_height;
			float win_xleft_low = leftx_cur - __window_half_width;
			float win_xleft_high = leftx_cur + __window_half_width;
			float win_xright_low = rightx_cur - __window_half_width;
			float win_xright_high = rightx_cur + __window_half_width;
			good_left_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xleft_low) & (nonzx <= win_xleft_high);
			good_right_inds = (nonzy >= win_y_low) & (nonzy <= win_y_high) & (nonzx >= win_xright_low) & (nonzx <= win_xright_high);
			
			left_lane_inds = left_lane_inds | good_left_inds;
			right_lane_inds = right_lane_inds | good_right_inds;
			
			if (countNonZero(good_left_inds) > __window_min_pixel)
			{
				Scalar leftx_curr = mean(nonzx, good_left_inds);
				leftx_cur = leftx_curr(0);
			}
			if (countNonZero(good_right_inds) > __window_min_pixel)
			{
				Scalar rightx_curr = mean(nonzx, good_right_inds);
				rightx_cur = rightx_curr(0);
			}
		}
	}
	else
	{
		left_lane_inds = (nonzx > (__left_fit[2]*((frow - 1 - nonzy).mul(frow - 1 - nonzy))+__left_fit[1]*(frow - 1 - nonzy) + __left_fit[0] - __window_half_width))
		& (nonzx < (__left_fit[2]*((frow - 1 - nonzy).mul(frow - 1 - nonzy))+__left_fit[1]*(frow - 1 - nonzy) + __left_fit[0] + __window_half_width));
		right_lane_inds = (nonzx > (__right_fit[2]*((frow - 1 - nonzy).mul(frow - 1 - nonzy))+__right_fit[1]*(frow - 1 - nonzy) + __right_fit[0] - __window_half_width))
		& (nonzx < (__right_fit[2]*((frow - 1 - nonzy).mul(frow - 1 - nonzy))+__right_fit[1]*(frow - 1 - nonzy) + __right_fit[0] + __window_half_width)); // formula's y is from downside
	}
	
	time_t t_temp3 = clock();
	cout << "Toarray: " << to_string(((float)(t_temp2 - t_temp1))/CLOCKS_PER_SEC) << "s. Window: ";
			cout << to_string(((float)(t_temp3 - t_temp2))/CLOCKS_PER_SEC) <<"s. Total: ";
			cout << to_string(((float)(t_temp3 - t_temp1))/CLOCKS_PER_SEC) <<"s. " << endl << endl;
	
	/// fit the interested pixels (start to loop)
	Mat left_fit, right_fit;
	int iteration_subsample_num = 3;
	
	
	for (int iteration_subsample = 0; iteration_subsample < iteration_subsample_num; iteration_subsample++)
	{
		time_t t_temp4 = clock();
		/// transform the pixels from valarray into Mat
		Mat left_indx, right_indx;
		findNonZero(left_lane_inds, left_indx);
		findNonZero(right_lane_inds, right_indx);

		//Mat left_seq = left_indx.col(0);
		//Mat right_seq = right_indx.col(0);
		left_indx = left_indx.reshape(1);
		right_indx = right_indx.reshape(1);
		
		Mat_<int> left_seq = (Mat_<int> ) left_indx.col(1);
		Mat_<int> right_seq = (Mat_<int> ) right_indx.col(1);
		
		time_t t_temp5 = clock();

		float length_left = left_indx.rows;
		float length_right = right_indx.rows;
		
		Mat X_left(1, length_left, CV_32F);
		Mat X_right(1, length_right, CV_32F);
		Mat Y_left(3, length_left, CV_32F, Scalar_<float>(1));
		Mat Y_right(3, length_right, CV_32F, Scalar_<float>(1));
		
		float* line_Yl1 = Y_left.ptr<float>(1);
		float* line_Yl2 = Y_left.ptr<float>(2);
		float* line_Xl = X_left.ptr<float>();
		float* line_Yr1 = Y_right.ptr<float>(1);
		float* line_Yr2 = Y_right.ptr<float>(2);
		float* line_Xr = X_right.ptr<float>();
		
		if (iteration_subsample == iteration_subsample_num - 1) // only in last iteration
		{
			__leftx.resize(length_left);
			__lefty.resize(length_left);
			__rightx.resize(length_right);
			__righty.resize(length_right);
			for (int i = 0; i < length_left; i++)
			{
				float cur_x = nonzx_(left_seq(i));
				float cur_y = nonzy_(left_seq(i));
				line_Yl1[i] = ( frow - 1 - cur_y );
				line_Yl2[i] = ( frow - 1 - cur_y )*( frow - 1 - cur_y ); // from downside
				line_Xl[i] = cur_x;
				__leftx[i] = cur_x;
				__lefty[i] = cur_y;
				__lane_window_out_img.at<Vec3b>(cur_y, cur_x) = Vec3b(0, 0, 150);
			}
			for (int i = 0; i < length_right; i++)
			{
				float cur_x = nonzx_(right_seq(i));
				float cur_y = nonzy_(right_seq(i));
				line_Yr1[i] = (frow - 1 - cur_y );
				line_Yr2[i] = (frow - 1 - cur_y )*(frow - 1 - cur_y );
				line_Xr[i] = cur_x;
				__rightx[i] = cur_x;
				__righty[i] = cur_y;
				__lane_window_out_img.at<Vec3b>(cur_y, cur_x) = Vec3b(0, 0, 150);
			}
			
			
			for (int i = 0; i < ite; i++)
			{
				left_lane_inds_vala[i] = (left_lane_inds.at<uchar>(i) != 0);
				right_lane_inds_vala[i] = (right_lane_inds.at<uchar>(i) != 0);
				nonzx_vala[i] = nonzx_(i);
				nonzy_vala[i] = nonzy_(i);
			}
		}
		else
		{
			for (int i = 0; i < length_left; i++)
			{
				float cur_x = nonzx_(left_seq(i));
				float cur_y = nonzy_(left_seq(i));
				line_Yl1[i] = ( frow - 1 - cur_y );
				line_Yl2[i] = ( frow - 1 - cur_y )*( frow - 1 - cur_y ); // from downside
				line_Xl[i] = cur_x;
			}
			for (int i = 0; i < length_right; i++)
			{
				float cur_x = nonzx_(right_seq(i));
				float cur_y = nonzy_(right_seq(i));
				line_Yr1[i] = (frow - 1 - cur_y );
				line_Yr2[i] = (frow - 1 - cur_y )*(frow - 1 - cur_y );
				line_Xr[i] = cur_x;
			}
		}
		
		time_t t_temp6 = clock();
		
		/// start robust regression
		left_fit = (Y_left.t()).inv(DECOMP_SVD)*(X_left.t());
		right_fit = (Y_right.t()).inv(DECOMP_SVD)*(X_right.t());
		
		
		if (iteration_subsample == iteration_subsample_num - 1) // last iteration is different
		{
			int iteration_num = 2;
			Mat res_left, res_right;
			for (int iteration = 0; iteration < iteration_num; iteration++)
			{
				res_left = (abs(X_left - left_fit.at<float>(2)*Y_left.row(2) - left_fit.at<float>(1)*Y_left.row(1) - left_fit.at<float>(0))).t() + 0.1;
				res_right = (abs(X_right - right_fit.at<float>(2)*Y_right.row(2) - right_fit.at<float>(1)*Y_right.row(1) - right_fit.at<float>(0))).t() + 0.1;
				
				res_left = 1/res_left + (frow - 1 - Y_left.row(1)).t()*0.1;
				res_right = 1/res_right + (frow - 1 - Y_right.row(1)).t()*0.1; // from downside
				Mat w_left(res_left.rows, 3, CV_32F);
				w_left.col(0) = res_left.col(0) + 0;
				w_left.col(1) = res_left.col(0) + 0;
				w_left.col(2) = res_left.col(0) + 0;
				Mat w_right(res_right.rows, 3, CV_32F);
				w_right.col(0) = res_right.col(0) + 0;
				w_right.col(1) = res_right.col(0) + 0;
				w_right.col(2) = res_right.col(0) + 0;
				left_fit = (Y_left.mul(w_left.t())*Y_left.t()).inv()*Y_left.mul(w_left.t())*X_left.t();
				right_fit = (Y_right.mul(w_right.t())*Y_right.t()).inv()*Y_right.mul(w_right.t())*X_right.t();
			}
			__left_fit = left_fit; // Mat(1,3) to Vec3f, OK???
			__right_fit = right_fit;
			
			
			/// also train a linear model
			Mat Y_left_lin = Y_left.rowRange(0, 2);
			Mat Y_right_lin = Y_right.rowRange(0, 2);
			Mat left_fit_2 = (Y_left_lin.t()).inv(DECOMP_SVD)*(X_left.t());
			Mat right_fit_2 = (Y_right_lin.t()).inv(DECOMP_SVD)*(X_right.t());
			__left_fit_2 = left_fit_2;
			__right_fit_2 = right_fit_2;
			

			__left_fit_cr[0] = __left_fit[0]*xm_per_pix;
			__left_fit_cr[1] = __left_fit[1]*xm_per_pix/ym_per_pix;
			__left_fit_cr[2] = __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			__right_fit_cr[0] = __right_fit[0]*xm_per_pix;
			__right_fit_cr[1] = __right_fit[1]*xm_per_pix/ym_per_pix;
			__right_fit_cr[2] = __right_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			
			
			/// train decision tree
			//trainmodel(warped_filter_image_U, nonzx, nonzy, left_lane_inds, right_lane_inds);
			trainmodel(warped_filter_image_U, nonzx_vala, nonzy_vala, left_lane_inds_vala, right_lane_inds_vala);
			
			
			//cout << "real formula from trans: [" << __left_fit[0]*xm_per_pix << ", " << __left_fit[1]*xm_per_pix/ym_per_pix << ", " << __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix << "]" << endl;
			// Mat(1,3) to Vec3f, OK???	
			break;
		}
		
		
		int iteration_num = 5;
		for (int iteration = 0; iteration < iteration_num; iteration++)
		{
			Mat res_left = (abs(X_left - left_fit.at<float>(2)*Y_left.row(2) - left_fit.at<float>(1)*Y_left.row(1) - left_fit.at<float>(0))).t() + 0.1;
			Mat res_right = (abs(X_right - right_fit.at<float>(2)*Y_right.row(2) - right_fit.at<float>(1)*Y_right.row(1) - right_fit.at<float>(0))).t() + 0.1;
			
			Mat w_left(res_left.rows, 3, CV_32F);
			w_left.col(0) = 5/res_left.col(0);
			w_left.col(1) = 5/res_left.col(0);
			w_left.col(2) = 5/res_left.col(0);
			Mat w_right(res_right.rows, 3, CV_32F);
			w_right.col(0) = 5/res_right.col(0);
			w_right.col(1) = 5/res_right.col(0);
			w_right.col(2) = 5/res_right.col(0);
			left_fit = (Y_left.mul(w_left.t())*Y_left.t()).inv()*Y_left.mul(w_left.t())*X_left.t();
			right_fit = (Y_right.mul(w_right.t())*Y_right.t()).inv()*Y_right.mul(w_right.t())*X_right.t();
		}
		time_t t_temp7 = clock();
		
		/// renew interested pixels
		float decay_coeff = 0.4;
		float left_fit_at[3] = {left_fit.at<float>(0), left_fit.at<float>(1), left_fit.at<float>(2)};
		float right_fit_at[3] = {right_fit.at<float>(0), right_fit.at<float>(1), right_fit.at<float>(2)};
		
		left_lane_inds = (nonzx - (left_fit_at[2]*(frow - 1 - nonzy).mul(frow - 1 - nonzy) + left_fit_at[1]*(frow - 1 - nonzy) + left_fit_at[0] - __window_half_width*decay_coeff) > 0 ) 
		& (nonzx -( left_fit_at[2]*(frow - 1 - nonzy).mul(frow - 1 - nonzy) + left_fit_at[1]*(frow - 1 - nonzy) + left_fit_at[0] + __window_half_width*decay_coeff) < 0 );
		
		right_lane_inds = (nonzx - ( right_fit_at[2]*(frow - 1 - nonzy).mul(frow - 1 - nonzy) + right_fit_at[1]*(frow - 1 - nonzy) + right_fit_at[0] - __window_half_width*decay_coeff) > 0 )
		& (nonzx -( right_fit_at[2]*(frow - 1 - nonzy).mul(frow - 1 - nonzy) + right_fit_at[1]*(frow - 1 - nonzy) + right_fit_at[0] + __window_half_width*decay_coeff) < 0);	
		
		
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

void Line::processNewRecord(bool van_consist)
{
	float w_current = 1 - w_history;
	if (current_fit.size() >= 1)
	{
		if (current_fit.size() > 2)
			cout << "diffs: " << abs(diffs[0])+abs(diffs[1])+abs(diffs[2]) << endl;
			
		if ( check > 0 && detected == false && van_consist == true) // history is bad and current one is good
		{
			best_fit = current_fit.back();
			fitRealWorldCurve();
			best_radius_of_curvature = radius_of_curvature.back();
			best_line_base_pos = line_base_pos.back();
			
			best_fit_2 = current_fit_2.back();
			
			fail_detect_count = 0;
			detected = true;
			cout << "case 1" << endl;
		}
		else if (check > 0 && current_fit.size() > 2 && abs(diffs[0])+abs(diffs[1])+abs(diffs[2]) < 20 && detected == true && van_consist == true) // history is good and current one is good  (modified: check should betrue)
		{
			best_fit = w_history*best_fit + w_current*current_fit.back();
			fitRealWorldCurve();
			//best_radius_of_curvature = w_history*best_radius_of_curvature + w_current*radius_of_curvature.back();
			//best_line_base_pos = w_history*best_line_base_pos + w_current*line_base_pos.back();
			best_radius_of_curvature = getCurvature();
			best_line_base_pos = getDistanceToLane();
			diffs = current_fit.back() - best_fit;
			
			best_fit_2 = w_history*best_fit_2 + w_current*current_fit_2.back();
			
			fail_detect_count = 0;
			cout << "case 2" << endl;
			
		}
		else if (check > 0 && current_fit.size() > 2 && abs(diffs[0])+abs(diffs[1])+abs(diffs[2]) < 100 && detected == true && van_consist == true) // history is soso(have fluctuation) and current one is good (modified: check should betrue)
		{
			w_current *= 0.6;
			best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
			fitRealWorldCurve();
			//best_radius_of_curvature = (1 - w_current)*best_radius_of_curvature + w_current*radius_of_curvature.back();
			//best_line_base_pos = (1 - w_current)*best_line_base_pos + w_current*line_base_pos.back();
			best_radius_of_curvature = getCurvature();
			best_line_base_pos = getDistanceToLane();
			diffs = current_fit.back() - best_fit;
			
			best_fit_2 = (1 - w_current)*best_fit_2 + w_current*current_fit_2.back();
			
			fail_detect_count = 0;
			cout << "case 3" << endl;
		}
		else // current one is bad
		{
			if (check > 0 || van_consist == true)
				w_current *= 0.2;
			else
				w_current = 0;
			best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
			fitRealWorldCurve();
			//best_radius_of_curvature = (1 - w_current)*best_radius_of_curvature + w_current*radius_of_curvature.back();
			//best_line_base_pos = (1 - w_current)*best_line_base_pos + w_current*line_base_pos.back();
			best_radius_of_curvature = getCurvature();
			best_line_base_pos = getDistanceToLane();
			if (check > 0 || van_consist == true) // else make no change to the Line class
				diffs = current_fit.back() - best_fit;
			
			best_fit_2 = (1 - w_current)*best_fit_2 + w_current*current_fit_2.back();
			
			fail_detect_count ++;
			if (fail_detect_count >= 5) detected = false;
			cout << "case 4" ;
			if (w_current == 0)
				cout << "b" << endl;
			else
				cout << "a" << endl;
		}
		__w_current = w_current;
	}
	return;
}


/// gradient filter considering dark-bright-dark
	Mat sobelx, sobely;
	Sobel(gray, sobelx, CV_32F, 1, 0, __sobel_kernel_size );
	Sobel(gray, sobely, CV_32F, 0, 1, __sobel_kernel_size );
	
	Mat sobel_xp, sobel_xn;
	threshold(sobelx, sobel_xp, 0, 1, THRESH_TOZERO);
	sobel_xn = -sobelx;
	threshold(sobel_xn, sobel_xn, 0, 1, THRESH_TOZERO);
	
	Mat filter_binary_x_p, filter_binary_x_n, filter_binary_dir;
	sobelAbsThresh(sobel_xp, filter_binary_x_p, __abs_x_thresh);
	sobelAbsThresh(sobel_xn, filter_binary_x_n, __abs_x_thresh);
	sobelDirThresh(sobelx, sobely, filter_binary_dir, __dir_thresh);
	
	time_t t_temp6 = clock();
	
	filter_binary_x_p = filter_binary_x_p & filter_binary_dir;
	filter_binary_x_n = filter_binary_x_n & filter_binary_dir;
	
	cout << "dilate width: " << __window_half_width/5 << endl;
	#ifndef NDEBUG_GR
	imshow("filter_binary_p", filter_binary_x_p);
	imshow("filter_binary_n", filter_binary_x_n);
	#endif
	Mat filter_binary_p_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	Mat filter_binary_n_move(filter_binary_x_p.size(), filter_binary_x_p.type(), Scalar(0));
	filter_binary_p_move.colRange((int)__window_half_width/10, warp_col) = filter_binary_x_p.colRange(0, warp_col - (int)__window_half_width/10) + 0;
	filter_binary_n_move.colRange(0, warp_col - (int)__window_half_width/10) = filter_binary_x_n.colRange((int)__window_half_width/10, warp_col) + 0;
	Mat expand_kernel = getStructuringElement(MORPH_RECT, Size(__window_half_width/5, 1));
	dilate(filter_binary_p_move, filter_binary_p_move, expand_kernel);
	dilate(filter_binary_n_move, filter_binary_n_move, expand_kernel);
	#ifndef NDEBUG_GR
	imshow("filter_binary_p_dilate", filter_binary_p_move);
	imshow("filter_binary_n_dilate", filter_binary_n_move);
	#endif
	Mat binary_output_gradient = filter_binary_p_move & filter_binary_n_move;
	binary_output_gradient.convertTo(binary_output_gradient, binary_output_color.type()); 
	#ifndef NDEBUG_GR
	imshow("binary_output_gradient", binary_output_gradient);
	waitKey(0);
	#endif
	
	/*
	Mat filter_binary_x, filter_binary_y, filter_binary_mag, filter_binary_dir;
	sobelAbsThresh(sobelx, filter_binary_x, __abs_x_thresh);
	sobelAbsThresh(sobely, filter_binary_y, __abs_y_thresh);
	sobelMagThresh(sobelx, sobely, filter_binary_mag, __mag_thresh);
	sobelDirThresh(sobelx, sobely, filter_binary_dir, __dir_thresh);
	
	time_t t_temp6 = clock();
	
	Mat binary_output_gradient;
	binary_output_gradient = filter_binary_x.mul(filter_binary_y) + filter_binary_mag.mul(filter_binary_dir);
	
	binary_output_gradient.setTo(0, warp_mask);
	
	minMaxLoc(binary_output_gradient, NULL, &max_val, NULL, NULL);
	binary_output_gradient = binary_output_gradient*(1/max_val);
	threshold(binary_output_gradient, binary_output_gradient, 0.5, 1, THRESH_TOZERO);
	*/



void sobelDirThresh(const Mat& sobelx, const Mat& sobely, Mat& binary_output, Vec2f thresh)
{
	Mat sobel_dir;
	phase(abs(sobelx), abs(sobely), sobel_dir); // not in degree
	//normalize(sobel_dir, sobel_dir, 0, 255, NORM_MINMAX ); // normalized to 0-255
	double max_val, min_val;
	minMaxLoc(sobel_dir, &min_val, &max_val, NULL, NULL);
	float thresh_l = thresh[0]/255*(max_val - min_val) + min_val;
	float thresh_h = thresh[1]/255*(max_val - min_val) + min_val;
	
	binary_output = Mat::ones(sobel_dir.size(), CV_32FC1);
	Mat mask = (sobel_dir <= thresh_l) | (sobel_dir >= thresh_h );
	binary_output.setTo(0.5, mask);
	
	#ifndef NDEBUG_GR
	imshow("sobel dir", mask);
	waitKey(0);
	#endif
	
	return;
}


void sobelAbsThresh(const Mat& sobel, Mat& binary_output, Vec2f thresh)
{
	// Mat abs_sobel;
	// threshold(abs_sobel, abs_sobel, 0, 1, THRESH_TOZERO);
	
	Mat abs_sobel = abs(sobel);
	
	//normalize(abs_sobel, abs_sobel, 0, 255, NORM_MINMAX );
	double max_val;
	minMaxLoc(abs_sobel, NULL, &max_val, NULL, NULL);
	float thresh_l = thresh[0]/255*max_val;
	float thresh_h = thresh[1]/255*max_val;
	

	binary_output = Mat::ones(abs_sobel.size(), CV_32FC1);
	Mat mask = (abs_sobel <= thresh_l) | (abs_sobel >= thresh_h);
	binary_output.setTo(0.5, mask);
	
	#ifndef NDEBUG_GR
	imshow("sobel abs", mask);
	waitKey(0);
	#endif
	
	return;
}

void sobelMagThresh(const Mat& sobelx, const Mat& sobely, Mat& binary_output, Vec2f thresh)
{
	Mat sobel_mag;
	magnitude(sobelx, sobely, sobel_mag);
	//normalize(sobel_mag, sobel_mag, 0, 255, NORM_MINMAX );
	double max_val, min_val;
	minMaxLoc(sobel_mag, &min_val, &max_val, NULL, NULL);
	float thresh_l = thresh[0]/255*(max_val - min_val) + min_val;
	float thresh_h = thresh[1]/255*(max_val - min_val) + min_val;
	
	binary_output = Mat::ones(sobel_mag.size(), CV_32FC1);
	Mat mask = (sobel_mag <= thresh_l) | (sobel_mag >= thresh_h);
	binary_output.setTo(0.5, mask);
	
	#ifndef NDEBUG_GR
	imshow("sobel mag", mask);
	//waitKey(0);
	#endif

	return;
}

void LaneImage::__reshapeSub(int half_width, const Mat& warp_reshape, const Mat& warp_reshape_HLS, Mat& warp_reshape_sub, Mat& warp_reshape_sub_HLS, vector<Point>& sub_pts )
{
	float frow = (float)warp_row;
	warp_reshape_sub = Mat(warp_row*half_width*4, warp_reshape.cols,  warp_reshape.type());
	warp_reshape_sub_HLS = Mat(warp_row*half_width*4, warp_reshape.cols,  warp_reshape.type());
	sub_pts.clear();
	sub_pts.reserve(warp_row*half_width*4);
	
	int num_of_rows = 0;
	for (int i = 0; i < warp_row; i++)
	{
		int left_min = max(__last_left_fit[2]*(frow - 1 - i)*(frow - 1 - i) + __last_left_fit[1]*(frow - 1 - i) + __last_left_fit[0] - half_width, (float)0);
		int left_max = min(left_min + 2*half_width, warp_col-1 );
		int right_min = max(__last_right_fit[2]*(frow - 1 - i)*(frow - 1 - i) + __last_right_fit[1]*(frow - 1 - i) + __last_right_fit[0] - half_width, (float)0);
		int right_max = min(right_min + 2*half_width, warp_col-1 );
		if (left_min <= warp_col-1 && left_max >= 0)
		for (int j = left_min; j < left_max; j++)
		{
			
			warp_reshape_sub.row(num_of_rows) = warp_reshape.row(i*warp_col + j) + 0;
			warp_reshape_sub_HLS.row(num_of_rows) = warp_reshape_HLS.row(i*warp_col + j) + 0;
			sub_pts.push_back(Point(j,i));
			num_of_rows ++;
			
			/*
			#ifdef DTREE
			__BGR_tree->predict(warp_reshape.row(i*warp_col + j), binary_output_white(Range(i,i+1),Range(j, j+1)) );
			__HLS_tree->predict(warp_reshape_HLS.row(i*warp_col + j), binary_output_yellow(Range(i,i+1),Range(j, j+1)) );
			#endif
			#ifdef LREGG
			__BGR_regg->predict(warp_reshape.row(i*warp_col + j), binary_output_white(Range(i,i+1),Range(j, j+1)) );
			__HLS_regg->predict(warp_reshape_HLS.row(i*warp_col + j), binary_output_yellow(Range(i,i+1),Range(j, j+1)) );
			#endif
			*/
		}
		if (right_min <= warp_col-1 &&right_max >= 0)
		for (int j = right_min; j < right_max; j++)
		{
			
			warp_reshape_sub.row(num_of_rows) = warp_reshape.row(i*warp_col + j) + 0;
			warp_reshape_sub_HLS.row(num_of_rows) = warp_reshape_HLS.row(i*warp_col + j) + 0;
			sub_pts.push_back(Point(j,i));
			num_of_rows ++;
			
			/*
			#ifdef DTREE
			__BGR_tree->predict(warp_reshape.row(i*warp_col + j), binary_output_white(Range(i,i+1),Range(j, j+1)) );
			__HLS_tree->predict(warp_reshape_HLS.row(i*warp_col + j), binary_output_yellow(Range(i,i+1),Range(j, j+1)) );
			#endif
			#ifdef LREGG
			__BGR_regg->predict(warp_reshape.row(i*warp_col + j), binary_output_white(Range(i,i+1),Range(j, j+1)) );
			__HLS_regg->predict(warp_reshape_HLS.row(i*warp_col + j), binary_output_yellow(Range(i,i+1),Range(j, j+1)) );
			#endif
			*/
		}
	}
	
	warp_reshape_sub.resize(num_of_rows);
	warp_reshape_sub_HLS.resize(num_of_rows);
	/*
	#ifdef LREGG
	binary_output_white.convertTo(binary_output_white, CV_32FC1);
	binary_output_yellow.convertTo(binary_output_yellow, CV_32FC1);
	#endif
	*/
	return;
}


// without sampling considering upper side and down side
void LaneImage::trainmodel(Mat& warped_filter_image_U, valarray<float>& nonzx, valarray<float>& nonzy, valarray<bool>& left_lane_inds, valarray<bool>& right_lane_inds)
{
	float frow = (float)warp_row;
			/// determine the size of samples of each categories
			time_t t_temp1 = clock();
			
			float length_left = __leftx.size();
			float length_right = __rightx.size();
			
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
			vector<Point> peak_loc_r, peak_loc_l;
			findNonZero(warped_filter_peak_r, peak_loc_r);
			findNonZero(warped_filter_peak_l, peak_loc_l);
			float length_peak_r = peak_loc_r.size();
			float length_peak_l = peak_loc_l.size();
			
			valarray<bool> non_lane_inds = !(left_lane_inds || right_lane_inds);
			valarray<float> notlane_x_pre = nonzx[ non_lane_inds ];
			valarray<float> notlane_y_pre = nonzy[ non_lane_inds ];
			
			valarray<bool> close_left_inds = (notlane_x_pre > __left_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __left_fit[1]*(frow - 1 - notlane_y_pre) + __left_fit[0] - __window_half_width) 
		& (notlane_x_pre < __left_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __left_fit[1]*(frow - 1 - notlane_y_pre) + __left_fit[0] + __window_half_width);
			valarray<bool> close_right_inds = (notlane_x_pre > __right_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __right_fit[1]*(frow - 1 - notlane_y_pre) + __right_fit[0] - __window_half_width) 
		& (notlane_x_pre < __right_fit[2]*(frow - 1 - notlane_y_pre)*(frow - 1 - notlane_y_pre) + __right_fit[1]*(frow - 1 - notlane_y_pre) + __right_fit[0] + __window_half_width);
			valarray<bool> close_lane_inds = close_left_inds || close_right_inds;
			
			valarray<float> notlane_x(notlane_x_pre[close_lane_inds]);
			valarray<float> notlane_y(notlane_y_pre[close_lane_inds]);
			
			
			float length_notlane = notlane_x.size();
			
			cout << "length_peak_l: " << length_peak_l << ", length_peak_r: " << length_peak_r << ", length_notlane: " << length_notlane << ", length_flip: " << length_flip << endl;
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
			int n_samp_pos_l, n_samp_pos_r; 
			if (__left_nolane)
			{
				n_samp_pos_l = 0;
				n_samp_pos_r = n_samp_pos;
			}
			else if (__right_nolane)
			{
				n_samp_pos_l = n_samp_pos;
				n_samp_pos_r = 0;
			}
			else
			{
				n_samp_pos_l = min( (float)0.8, max(length_left/ (length_left + length_right), (float) 0.2) ) *n_samp_pos;
				n_samp_pos_r = n_samp_pos - n_samp_pos_l; 
			}
				
			int num_left, num_right, num_notlane, num_flip, num_lane;
			num_left = n_samp_pos_l;
			num_right = n_samp_pos_r;
			num_notlane = (int)(min(  length_notlane,  max((float)0.05, length_notlane/ (length_notlane + length_flip))* n_samp_neg  ));
			num_flip = n_samp - num_left-num_right-num_notlane ; // need change, how to handle extreme cases when sample points are few
			num_lane = num_left + num_right;
			cout << "n_sample: " << num_left <<" " << num_right <<" " << num_lane << " " << num_notlane <<" " << num_flip << endl;
			
			
			time_t t_temp2;
			if ( __last_left_fit == Vec3f(0, 0, 0) || __last_right_fit == Vec3f(0, 0, 0) ) // __nframe == 0
			{
				int mode_for_pos_samp; // for indicating whether peak samples are enough
				if ( (length_peak_l > 5 * n_samp_pos_l || __left_nolane) && (length_peak_r > 5 * n_samp_pos_r || __right_nolane) )
					mode_for_pos_samp = 1;
				else
					mode_for_pos_samp = 0;
				cout << "mode_for_pos_samp: " << mode_for_pos_samp << endl;
				
				
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
					
					if (mode_for_pos_samp == 0)
					{
						if (length_left != 0 && num_left != 0 )
						for (int i = 0; i < num_left; i++)
						{
							ind = rand() % (int)length_left;
							__BGR_sample.row(i + j * n_samp) = __warped_reshape.row(__lefty[ind]*warp_col + __leftx[ind]) + 0;
							__HLS_sample.row(i + j * n_samp) = __warped_reshape_HLS.row(__lefty[ind]*warp_col + __leftx[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(__lefty[ind], __leftx[ind]) *= 2;
						}
						if (length_right != 0 && num_right != 0 )
						for (int i = 0; i < num_right; i++)
						{
							ind = rand() % (int)length_right;
							__BGR_sample.row(i + num_left + j * n_samp) = __warped_reshape.row(__righty[ind]*warp_col + __rightx[ind]) + 0;
							__HLS_sample.row(i + num_left + j * n_samp) = __warped_reshape_HLS.row(__righty[ind]*warp_col + __rightx[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(__righty[ind], __rightx[ind]) *= 2;
						}
					}
					else
					{
						if (length_peak_l != 0 && num_left != 0 )
						for (int i = 0; i < num_left; i++)
						{
							ind = rand() % (int)length_peak_l;
							__BGR_sample.row(i + j * n_samp) = __warped_reshape.row(peak_loc_l[ind].y*warp_col + peak_loc_l[ind].x) + 0;
							__HLS_sample.row(i + j * n_samp) = __warped_reshape_HLS.row(peak_loc_l[ind].y*warp_col + peak_loc_l[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_l[ind] ) *= 2;
						}
						if (length_peak_r != 0 && num_right != 0 )
						for (int i = 0; i < num_right; i++)
						{
							ind = rand() % (int)length_peak_r;
							__BGR_sample.row(i + num_left + j * n_samp) = __warped_reshape.row(peak_loc_r[ind].y*warp_col + peak_loc_r[ind].x) + 0;
							__HLS_sample.row(i + num_left + j * n_samp) = __warped_reshape_HLS.row(peak_loc_r[ind].y*warp_col + peak_loc_r[ind].x) + 0;
							__lane_window_out_img.at<Vec3b>( peak_loc_r[ind] ) *= 2;
						}
					}
					if (length_notlane != 0 && num_notlane != 0)
						for (int i = 0; i < num_notlane; i++)
						{
							ind = rand() % (int)length_notlane;
							__BGR_sample.row(i + n_samp_pos + j * n_samp) = __warped_reshape.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
							__HLS_sample.row(i + n_samp_pos + j * n_samp) = __warped_reshape_HLS.row(notlane_y[ind]*warp_col + notlane_x[ind]) + 0;
							__lane_window_out_img.at<Vec3b>(notlane_y[ind], notlane_x[ind]) = Vec3b(0, 255, 0);
						}
					
					for (int i = 0; i < num_flip; i++)
					{
						ind = rand() % (int)length_flip;
						__BGR_sample.row(i + num_notlane +  n_samp_pos + j * n_samp) = __warped_reshape.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
						__HLS_sample.row(i + num_notlane +  n_samp_pos + j * n_samp) = __warped_reshape_HLS.row(zero_loc[ind].y*warp_col + zero_loc[ind].x) + 0;
						__lane_window_out_img.at<Vec3b>( zero_loc[ind] ) = Vec3b(0, 150, 0);
					}
				}
			}
			else
			{				
				int mode_for_pos_samp; // for indicating whether peak samples are enough
				if ( (length_peak_l > n_samp_pos_l || __left_nolane) && (length_peak_r > n_samp_pos_r || __right_nolane) )
					mode_for_pos_samp = 1;
				else
					mode_for_pos_samp = 0;
				cout << "mode_for_pos_samp: " << mode_for_pos_samp << endl;			
			
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
				if (mode_for_pos_samp == 0)
				{
					if (length_left != 0 && num_left != 0 )
					for (int i = 0; i < num_left; i++)
					{
						ind = rand() % (int)length_left;
						__BGR_sample.row(i + __samp_cyc * n_samp) = __warped_reshape.row(__lefty[ind]*warp_col + __leftx[ind]) + 0;
						__HLS_sample.row(i + __samp_cyc * n_samp) = __warped_reshape_HLS.row(__lefty[ind]*warp_col + __leftx[ind]) + 0;
						__lane_window_out_img.at<Vec3b>(__lefty[ind], __leftx[ind]) *= 2;
					}
					if (length_right != 0 && num_right != 0 )
					for (int i = 0; i < num_right; i++)
					{
						ind = rand() % (int)length_right;
						__BGR_sample.row(i + num_left + __samp_cyc * n_samp) = __warped_reshape.row(__righty[ind]*warp_col + __rightx[ind]) + 0;
						__HLS_sample.row(i + num_left + __samp_cyc * n_samp) = __warped_reshape_HLS.row(__righty[ind]*warp_col + __rightx[ind]) + 0;
						__lane_window_out_img.at<Vec3b>(__righty[ind], __rightx[ind]) *= 2;
					}
				}
				else
				{
					if (length_peak_l != 0 && num_left != 0 )
					for (int i = 0; i < num_left; i++)
					{
						ind = rand() % (int)length_peak_l;
						__BGR_sample.row(i + __samp_cyc * n_samp) = __warped_reshape.row(peak_loc_l[ind].y*warp_col + peak_loc_l[ind].x) + 0;
						__HLS_sample.row(i + __samp_cyc * n_samp) = __warped_reshape_HLS.row(peak_loc_l[ind].y*warp_col + peak_loc_l[ind].x) + 0;
						__lane_window_out_img.at<Vec3b>( peak_loc_l[ind] ) *= 2;
					}
					if (length_peak_r != 0 && num_right != 0 )
					for (int i = 0; i < num_right; i++)
					{
						ind = rand() % (int)length_peak_r;
						__BGR_sample.row(i + num_left + __samp_cyc * n_samp) = __warped_reshape.row(peak_loc_r[ind].y*warp_col + peak_loc_r[ind].x) + 0;
						__HLS_sample.row(i + num_left + __samp_cyc * n_samp) = __warped_reshape_HLS.row(peak_loc_r[ind].y*warp_col + peak_loc_r[ind].x) + 0;
						__lane_window_out_img.at<Vec3b>( peak_loc_r[ind] ) *= 2;
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
			
			cout << "left lane: " << length_left << ", right lane: " << length_right << ", nonlane: " << length_notlane << ", zeroloc: " << length_flip << endl;
			cout << "total: " << length_notlane + length_flip + length_left+length_right << ". real: " << warp_row*warp_col << endl;
			
			return;
}

// using new avg van
void renewSeriesVan(Point new_van_pt, vector<Point>& series_van_pt, Point& avg_series_van, int& pos_of_renew, bool& consist, float track_radius ) // process the van_pts that are not averaged throgh frames
{
	//float size_ref = img_size.width > img_size.height ? img_size.height : img_size.width;
	if ( series_van_pt.empty() )
	{
		series_van_pt.push_back(new_van_pt);
		avg_series_van = new_van_pt;
		return;
	}
	else 
	{
		vector<Point>::iterator first = series_van_pt.begin();
		vector<Point>::iterator last = series_van_pt.end();
		if (last - first == 1)
		{
			avg_series_van = *first;
			if (abs(new_van_pt.x - first->x) <= track_radius && abs(new_van_pt.y - first->y) <= track_radius )
			{
				series_van_pt.push_back(new_van_pt);
				avg_series_van = 0.5*(avg_series_van + new_van_pt);
			}
			else
			{
				*first = new_van_pt;
				avg_series_van = new_van_pt;
			}
		}
		else if (last - first < 5)
		{
			/*
			float x_avg = 0, y_avg = 0;
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				x_avg += it->x;
				y_avg += it->y;
			}
			x_avg = x_avg / (last-first);
			y_avg = y_avg / (last-first);
			avg_series_van.x = (int)x_avg;
			avg_series_van.y = (int)y_avg;
			*/
			if ( abs(new_van_pt.x - avg_series_van.x) <= track_radius && abs(new_van_pt.y - avg_series_van.y) <= track_radius)
			{
				series_van_pt.push_back(new_van_pt);
				avg_series_van = (1.0/(last - first + 1))*(((int)(last - first))*avg_series_van + new_van_pt);
			}
			/*
			series_van_pt.push_back(new_van_pt);
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				if ( abs(new_van_pt.x - it->x) > 20 || abs(new_van_pt.y - it->y) > 20 )
				{
					series_van_pt.pop_back();
					break;
				}
			}
			*/
		}
		else if (last - first == 5)
		{
			/*
			float x_avg = 0, y_avg = 0;
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				x_avg += it->x;
				y_avg += it->y;
			}
			x_avg = x_avg / (last-first);
			y_avg = y_avg / (last-first);
			avg_series_van.x = (int)x_avg;
			avg_series_van.y = (int)y_avg;
			*/
			if ( abs(new_van_pt.x - avg_series_van.x) <= track_radius && abs(new_van_pt.y - avg_series_van.y) <= track_radius) // temporally not using the relation with last point
			{
				consist = true;
				avg_series_van = (5*avg_series_van - series_van_pt[pos_of_renew]+ new_van_pt)*0.2;
				series_van_pt[pos_of_renew] = new_van_pt;
				pos_of_renew = (pos_of_renew + 1) % 5;
			}
			else
				consist = false;
			/*
			for (vector<Point>::iterator it = first; it != last; it++)
			{
				if ( abs(new_van_pt.x - it->x) > 20 || abs(new_van_pt.y - it->y) > 20 )
				{
					consist = false;
					break;
				}
			}
			if ( abs(new_van_pt.x - series_van_pt[ (pos_of_renew+4)%5 ].x) > 10 || abs(new_van_pt.y - series_van_pt[ (pos_of_renew+4)%5 ].y) > 10 )
			{
				consist = false;
			}
			
			if (consist)
			{
				series_van_pt[pos_of_renew] = new_van_pt;
				pos_of_renew = (pos_of_renew + 1) % 5;
			}
			*/
			cout << "pos_of_renew: " << pos_of_renew << ", consist: " << consist << endl;
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

// not using one model for two lanes
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
	Mat left_fit, right_fit;
	int iteration_subsample_num = 3;
	
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
		
		float length_left = leftx.size();
		float length_right = rightx.size();
		
		Mat X_left(1, length_left, CV_32F);
		Mat X_right(1, length_right, CV_32F);
		Mat Y_left(3, length_left, CV_32F, Scalar_<float>(1));
		Mat Y_right(3, length_right, CV_32F, Scalar_<float>(1));
		Mat W_left(1, length_left, CV_32F);
		Mat W_right(1, length_right, CV_32F);
		
		float* line_Yl1 = Y_left.ptr<float>(1);
		float* line_Yl2 = Y_left.ptr<float>(2);
		float* line_Xl = X_left.ptr<float>();
		float* line_Yr1 = Y_right.ptr<float>(1);
		float* line_Yr2 = Y_right.ptr<float>(2);
		float* line_Xr = X_right.ptr<float>();
		
		float* line_Wl = W_left.ptr<float>();
		float* line_Wr = W_right.ptr<float>();
		
		
		if (iteration_subsample == iteration_subsample_num - 1) // only in last iteration
		{
			for (int i = 0; i < length_left; i++)
			{
				line_Yl1[i] = (frow - 1 - lefty[i]);
				line_Yl2[i] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]); // small from downside
				line_Xl[i] = leftx[i];
				line_Wl[i] = 1.0/( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
				//line_Wl[i] = 1;
				__lane_window_out_img.at<Vec3b>(lefty[i], leftx[i]) = Vec3b(150, 0, 0);
			}
			for (int i = 0; i < length_right; i++)
			{
				line_Yr1[i] = (frow - 1 -righty[i]);
				line_Yr2[i] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
				line_Xr[i] = rightx[i];
				line_Wr[i] = 1.0/( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
				//line_Wr[i] = 1;
				__lane_window_out_img.at<Vec3b>(righty[i], rightx[i]) = Vec3b(0, 0, 150);
			}
		}
		else
		{
			for (int i = 0; i < length_left; i++)
			{
				line_Yl1[i] = (frow - 1 - lefty[i]);
				line_Yl2[i] = (frow - 1 - lefty[i])*(frow - 1 - lefty[i]);
				line_Xl[i] = leftx[i];
				line_Wl[i] = 1.0/( __warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i])*__warped_filter_image.at<float>(lefty[i], leftx[i]) );
				//line_Wl[i] = 1;
			}
			for (int i = 0; i < length_right; i++)
			{
				line_Yr1[i] = (frow - 1 -righty[i]);
				line_Yr2[i] = (frow - 1 -righty[i])*(frow - 1 -righty[i]);
				line_Xr[i] = rightx[i];
				line_Wr[i] =1.0/ ( __warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i])*__warped_filter_image.at<float>(righty[i], rightx[i]) );
				//line_Wr[i] = 1;
			}
		}
		
		time_t t_temp6 = clock();
		
		/// start robust regression
		left_fit = (Y_left.t()).inv(DECOMP_SVD)*(X_left.t());
		right_fit = (Y_right.t()).inv(DECOMP_SVD)*(X_right.t());
		
	
		if (iteration_subsample == iteration_subsample_num - 1) // last iteration is different
		{
			int iteration_num = 2;
			Mat res_left, res_right;
			for (int iteration = 0; iteration < iteration_num; iteration++)
			{
				res_left = (abs(X_left - left_fit.at<float>(2)*Y_left.row(2) - left_fit.at<float>(1)*Y_left.row(1) - left_fit.at<float>(0))).t() + 0.1;
				res_right = (abs(X_right - right_fit.at<float>(2)*Y_right.row(2) - right_fit.at<float>(1)*Y_right.row(1) - right_fit.at<float>(0))).t() + 0.1;
				
				res_left = 1/res_left + (frow - 1 - Y_left.row(1)).t()*0.1; //(frow - 1 - Y_left.row(1))
				res_right = 1/res_right + (frow - 1 - Y_right.row(1)).t()*0.1; // large for downside (frow - 1 - Y_right.row(1))
				Mat w_left(res_left.rows, 3, CV_32F);
				w_left.col(0) = res_left.col(0).mul(W_left.t());
				w_left.col(1) = res_left.col(0).mul(W_left.t());
				w_left.col(2) = res_left.col(0).mul(W_left.t());
				Mat w_right(res_right.rows, 3, CV_32F);
				w_right.col(0) = res_right.col(0).mul(W_right.t());
				w_right.col(1) = res_right.col(0).mul(W_right.t());
				w_right.col(2) = res_right.col(0).mul(W_right.t());
				left_fit = (Y_left.mul(w_left.t())*Y_left.t()).inv()*Y_left.mul(w_left.t())*X_left.t();
				right_fit = (Y_right.mul(w_right.t())*Y_right.t()).inv()*Y_right.mul(w_right.t())*X_right.t();
			}
			__left_fit = left_fit; // Mat(1,3) to Vec3f, OK???
			__right_fit = right_fit;
			__leftx = leftx;
			__lefty = lefty;
			__rightx = rightx;
			__righty = righty;
			
			/// also train a linear model
			Mat Y_left_lin = Y_left.rowRange(0, 2);
			Mat Y_right_lin = Y_right.rowRange(0, 2);
			Mat left_fit_2 = (Y_left_lin.t()).inv(DECOMP_SVD)*(X_left.t());
			Mat right_fit_2 = (Y_right_lin.t()).inv(DECOMP_SVD)*(X_right.t());
			__left_fit_2 = left_fit_2;
			__right_fit_2 = right_fit_2;
			

			__left_fit_cr[0] = __left_fit[0]*xm_per_pix;
			__left_fit_cr[1] = __left_fit[1]*xm_per_pix/ym_per_pix;
			__left_fit_cr[2] = __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			__right_fit_cr[0] = __right_fit[0]*xm_per_pix;
			__right_fit_cr[1] = __right_fit[1]*xm_per_pix/ym_per_pix;
			__right_fit_cr[2] = __right_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
			
			
			/// train decision tree
			__laneSanityCheck(hist_width);
			
			if ( (__parallel_check && __width_check) || __nframe < 5)
				trainmodel(warped_filter_image_U, nonzx, nonzy, left_lane_inds, right_lane_inds);
			

			//cout << "real formula from trans: [" << __left_fit[0]*xm_per_pix << ", " << __left_fit[1]*xm_per_pix/ym_per_pix << ", " << __left_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix << "]" << endl;
			// Mat(1,3) to Vec3f, OK???	
			break;
		}
		
		
		int iteration_num = 5;
		for (int iteration = 0; iteration < iteration_num; iteration++)
		{
			Mat res_left = (abs(X_left - left_fit.at<float>(2)*Y_left.row(2) - left_fit.at<float>(1)*Y_left.row(1) - left_fit.at<float>(0))).t() + 0.1;
			Mat res_right = (abs(X_right - right_fit.at<float>(2)*Y_right.row(2) - right_fit.at<float>(1)*Y_right.row(1) - right_fit.at<float>(0))).t() + 0.1;
			
			Mat w_left(res_left.rows, 3, CV_32F);
			w_left.col(0) = 5/res_left.col(0).mul(W_left.t());
			w_left.col(1) = 5/res_left.col(0).mul(W_left.t());
			w_left.col(2) = 5/res_left.col(0).mul(W_left.t());
			Mat w_right(res_right.rows, 3, CV_32F);
			w_right.col(0) = 5/res_right.col(0).mul(W_right.t());
			w_right.col(1) = 5/res_right.col(0).mul(W_right.t());
			w_right.col(2) = 5/res_right.col(0).mul(W_right.t());
			left_fit = (Y_left.mul(w_left.t())*Y_left.t()).inv()*Y_left.mul(w_left.t())*X_left.t();
			right_fit = (Y_right.mul(w_right.t())*Y_right.t()).inv()*Y_right.mul(w_right.t())*X_right.t();
		}
		time_t t_temp7 = clock();
		
		/// renew interested pixels
		float decay_coeff = 0.4; // 0.4
		float left_fit_at[3] = {left_fit.at<float>(0), left_fit.at<float>(1), left_fit.at<float>(2)};
		float right_fit_at[3] = {right_fit.at<float>(0), right_fit.at<float>(1), right_fit.at<float>(2)};
		
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


	int ksize = 49;
	double sigma = 13;
	Mat steer_resp_mag(image.size(), CV_32FC1);
	SteerFilter(image, steer_resp_mag, sobel_angle, ksize, sigma);
	



	
	#ifndef HIGH_BOT
	edges.rowRange(0, image.rows/2) = 0;
	#else
	edges.rowRange(0, image.rows/2) = 0;
	edges.rowRange(image.rows*7/10, image.rows) = 0;  // for caltech data
	#endif
	
	if (sucs_before)
	{
		edges.rowRange(0, warp_test_vec[0][0].y) = 0;
		edges.rowRange(warp_test_vec[0][1].y, image.rows) = 0;
		
	}
	
	
	
	
	
	
	
	
	/// vote for vanishing point based on Hough
	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI/180, 10, 10, 10 );
	//HoughLinesP(edges, lines, 20, CV_PI/180*3, 30, 10, 50 );
	
	if (lines.size() <= 0) /// safe return
	{
        fail_ini_count++; 
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
	
	float van_avg_x = van_pt_avg.x, van_avg_y = van_pt_avg.y;
	
	#ifndef NDEBUG_IN   // draw van_pt_avg tracking rectangle
	Mat edges_w_hist_van;
	edges.copyTo(edges_w_hist_van);
	if (series_van_pt.size() == 5) 
		rectangle(edges_w_hist_van, Point(image.cols/4, van_pt_avg.y - VAN_TRACK_Y*2), Point(image.cols*3/4, van_pt_avg.y + VAN_TRACK_Y*2), Scalar(255), -1);
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
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > image.cols - 1 || x_cur < 0)
					break;
				vote_right.at<double>(y_cur, x_cur)+= w;
			}
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
			for (int j = 0; j < y0 ; j++ )
			{
				int x_cur = x0 - k*j;
				int y_cur = y0 - j;
				if (x_cur > image.cols - 1 || x_cur < 0)
					break;
				vote_left.at<double>(y_cur, x_cur)+= w;
			}
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
	
	
	if (series_van_pt.size() == 5)
	{
		Mat mask_track(image.rows, image.cols, CV_8UC1, Scalar(255));
		rectangle(mask_track, Point(van_pt_avg.x - VAN_TRACK_Y*4, van_pt_avg.y - VAN_TRACK_Y*2), Point(van_pt_avg.x + VAN_TRACK_Y*4, van_pt_avg.y + VAN_TRACK_Y*2), Scalar(0), -1);
		votemap.setTo(0, mask_track); // remove the possible vanishing point out of the track radius*2
	}
	
	double maxval;
	Point van_pt_vote;
    minMaxLoc(votemap, NULL, &maxval, NULL, &van_pt_vote);
    if (maxval > 0)
    {
        van_pt_ini.x = van_pt_vote.x;
        van_pt_ini.y = van_pt_vote.y;
        cout << "first van pt: " << van_pt_ini << "maxval: "<< maxval << endl;
        fail_ini_count = 0;
    }
	else /// safe return
	{
        fail_ini_count++; 
		cout << "Initilization failed: no vanishing point found. " << endl;
		return -1;
	}
	
	#ifndef NDEBUG_IN
    /// draw the vote map
    imshow("vote left", vote_left);
	imshow("vote right", vote_right);
	Mat votemap_see;
	votemap.convertTo(votemap_see, CV_8U, 255.0/maxval, 0);
	circle(votemap_see, Point(van_pt_ini), 5, Scalar(255), -1);
	imshow("votemap", votemap_see);
	//circle(vote_line, van_pt_int, 5, Scalar(255), -1);
	imshow("vote_line", vote_line);
	waitKey(0);
	#endif
	
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
	
	/// generate thresholds for color channels
	Mat pixhis_b, pixhis_g, pixhis_r;
	Mat pixhis_h, pixhis_l, pixhis_s;
	Mat img_hls(color_img.size(), CV_8UC3);
	cvtColor(color_img, img_hls, COLOR_BGR2HLS);
	vector<Mat> bgr_chnl, hls_chnl;
	split(color_img, bgr_chnl);
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
	
	if (!sucs_before)
	{
		sucs_before = true;
		first_sucs = true;
	}

	return 0;
}