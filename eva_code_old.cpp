///// EVA module before 2018/02/15
///// According to this code, we import ground truth from MATLAB and do evaluation in c when the detection program is running. 
///// We don't want to do it now because the outputing result from c++ and do evaluation in MATLAB would be more straight forward. 
///// We numbered the four paragraphs of code and placed corresponding number in original code for possible recovering. 

#ifdef EVA // 1st paragraph
	/// for evaluate
	vector<float> out_rate;
	vector<float> error_rate;
	float out_rate_cur, error_accu;
	float fail_count = 0, fail_rate;
	float tp_count = 0, pos_count = 0; // number of frames that output a result
	float fp_count = 0, neg_count = 0;
	double truth_y[5];
	ifstream pt_file[5];
	for (int i = 0; i < 5; i++)
	{
		char pt_file_name[50];
		sprintf(pt_file_name,"row%d.txt",i+1);
		pt_file[i].open(pt_file_name);
		if (!pt_file[i])
		{
			cout << "Error: Open file " << pt_file_name << endl;
			return -1;
		}
		string lineread;
		getline(pt_file[i], lineread);
		sscanf(lineread.c_str(), "%lf", truth_y + i);
		//getline(pt_file[i], lineread);
	}
#endif

#ifdef EVA // 2nd paragraph
				/// evaluate the result quantitatively
				//if ( (time_step >= 46 && time_step <= 55) || (time_step >= 123 && time_step <= 145) || (time_step >= 214 && time_step <= 224) || (time_step >= 343 && time_step <= 362) )
				if ( (time_step >= 70 && time_step <= 79) || (time_step >= 133 && time_step <= 138) || (time_step >= 242 && time_step <= 246) )
				{ // false positive
					out_rate_cur = 0;
					error_accu = 0;
					
					fp_count ++;
					neg_count ++;
				}
				//else if ((time_step >= 39 && time_step <= 45) || (time_step >= 56 && time_step <= 57) ||(time_step >= 112 && time_step <= 122) ||
				//		(time_step >= 146 && time_step <= 152) ||(time_step >= 210 && time_step <= 213) ||(time_step >= 338 && time_step <= 342) ||(time_step >= 363 && time_step <= 365) )
				else if((time_step >= 62 && time_step <= 69) || (time_step >= 126 && time_step <= 132) || (time_step >= 139 && time_step <= 139) || (time_step >= 229 && time_step <= 241) )
				{ // not counted as positive or negative (just ignore)
					//string txtline;
					//for (int i = 0; i < 5; i++)
					//{
						//getline(pt_file[i],txtline);
					//}
					out_rate_cur = 0;
					error_accu = 0;
				}
				else
				{ // true positive
					Mat warp_zero_lr(Size(warp_col, warp_row), CV_8UC1, Scalar(0));
					vector<vector<Point> > plot_pts_vec_lr;
					plot_pts_vec_lr.push_back(plot_pts_l);
					plot_pts_vec_lr.push_back(plot_pts_r);
					polylines(warp_zero_lr, plot_pts_vec_lr, false, Scalar(255) );
					
					vector<Point> result_nzo;
					Mat newwap_binary;
					warpPerspective(warp_zero_lr, newwap_binary, van_pt.inv_per_mtx, image.size() );
					findNonZero(newwap_binary, result_nzo);
					
					//imshow("binary bird", warp_zero_lr);
					//imshow("binary lane", newwap_binary);
					double truth_x_left[5], truth_x_right[5];
					string txtline;
					for (int i = 0; i < 5; i++)
					{
						getline(pt_file[i],txtline);
						sscanf(txtline.c_str(), "%lf %lf", truth_x_left+i, truth_x_right+i);
						circle(newwarp, Point(truth_x_left[i], truth_y[i]), 7, Scalar(0, 0, 255), -1 );
						circle(newwarp, Point(truth_x_right[i], truth_y[i]), 7, Scalar(0, 0, 255), -1 ); // 3
					}
					
					//cout << "truth x left: " << truth_x_left[0] << truth_x_left[1] << truth_x_left[2] << truth_x_left[3] << truth_x_left[4] << endl;
					//cout << "truth x right: " << truth_x_right[0] << truth_x_right[1] << truth_x_right[2] << truth_x_right[3] << truth_x_right[4] << endl;
					//cout << "truth y: " << truth_y[0] << truth_y[1] << truth_y[2] << truth_y[3] << truth_y[4] << endl;
					
					//BSpline<float> spline_l(truth_y, 5, truth_x_left, 0);
					//BSpline<float> spline_r(truth_y, 5, truth_x_right, 0);
					alglib::real_1d_array truth_xx_left;
					alglib::real_1d_array truth_xx_right;
					alglib::real_1d_array truth_yy;
					truth_xx_left.setcontent(5, truth_x_left);
					truth_xx_right.setcontent(5, truth_x_right);
					truth_yy.setcontent(5, truth_y);
					
					alglib::spline1dinterpolant spline_l, spline_r;
					alglib::spline1dbuildcubic(truth_yy, truth_xx_left, spline_l);
					alglib::spline1dbuildcubic(truth_yy, truth_xx_right, spline_r);				
						int out_count = 0; 
						error_accu = 0;
						for (int i = 0; i < result_nzo.size(); i++)
						{
							//float x_eva_l = spline_l.evaluate((float)(result_nzo[i].y));
							//float x_eva_r = spline_r.evaluate((float)(result_nzo[i].y));
							double x_eva_l = alglib::spline1dcalc(spline_l, result_nzo[i].y);
							double x_eva_r = alglib::spline1dcalc(spline_r, result_nzo[i].y);
							
							if ( abs(x_eva_l - result_nzo[i].x) < abs(x_eva_r - result_nzo[i].x) )
							{
								circle(newwarp, Point(x_eva_l, result_nzo[i].y), 4, Scalar(0, 0, 255), -1 ); // 1
								//cout << "Left: " << x_eva_l << ", detect: " << result_nzo[i].y << endl;
								//cout << "Error: " << abs(x_eva_l - result_nzo[i].x)  << ", thresh: " << (3.0686*result_nzo[i].y - 1289.6)*(0.2+0.1*(truth_y[4]-result_nzo[i].y)/(truth_y[4]-truth_y[0]) )<< endl;
								float error_cur = abs(x_eva_l - result_nzo[i].x)/(1.4876*result_nzo[i].y - 256.6);
								error_accu += error_cur;
								if ( error_cur >  0.1 ) //(0.1+0.05*(truth_y[4]-result_nzo[i].y)/(truth_y[4]-truth_y[0]) )  )
								{
									out_count++;
								}
							}
							else
							{
								circle(newwarp, Point(x_eva_r, result_nzo[i].y), 4, Scalar(0, 0, 255), -1 ); // 1
								//cout << "Right: " << x_eva_r << ", detect: " << result_nzo[i].y << endl;
								//cout << "Error: " << abs(x_eva_r - result_nzo[i].x)  << ", thresh: " << (3.0686*result_nzo[i].y - 1289.6)*(0.2+0.1*(truth_y[4]-result_nzo[i].y)/(truth_y[4]-truth_y[0]) )<< endl;
								float error_cur = abs(x_eva_r - result_nzo[i].x)/(1.4876*result_nzo[i].y - 256.6);
								error_accu += error_cur;
								if ( error_cur > 0.1 )//(0.1+0.05*(truth_y[4]-result_nzo[i].y)/(truth_y[4]-truth_y[0]) )  )
								{
									out_count++;
								}
							}
						}
						out_rate_cur = (float)out_count / result_nzo.size();
						out_rate.push_back(out_rate_cur);
						error_accu = error_accu / result_nzo.size();
						error_rate.push_back(error_accu);
						if (out_rate_cur > 0.2)
							fail_count++;
						tp_count ++;
						pos_count++;
					cout << "out_rate: " << out_rate_cur << endl;
				}
				
				cout << "fail frames: " << fail_count <<  ", total frames: " << tp_count << endl;
				/// ////////////////////////////////////////////////////////////////////////////
#endif

#ifdef EVA // 3rd paragraph
			else
			{ // true or false negative
				out_rate_cur = 0;
				error_accu = 0;
				//if ( (time_step >= 46 && time_step <= 55) || (time_step >= 123 && time_step <= 145) || (time_step >= 214 && time_step <= 224) || (time_step >= 343 && time_step <= 362) )
				if ( (time_step >= 70 && time_step <= 79) || (time_step >= 133 && time_step <= 138) || (time_step >= 242 && time_step <= 246) )
				{
					neg_count ++;
				}
				else if((time_step >= 62 && time_step <= 69) || (time_step >= 126 && time_step <= 132) || (time_step >= 139 && time_step <= 139) || (time_step >= 229 && time_step <= 241) )
				{
					//string txtline;
					//for (int i = 0; i < 5; i++)
					//{
						//getline(pt_file[i],txtline);
					//}
				}
				else
				{
					pos_count++;
					string txtline;
					for (int i = 0; i < 5; i++)
					{
						getline(pt_file[i],txtline);
					}
				}
			}
#endif

#ifdef EVA // 4th paragraph
			float total_error = 0;
			float total_out = 0;
			for (int i = 0; i < error_rate.size(); i++)
			{
				total_error += error_rate[i];
			}
			for (int i = 0; i < out_rate.size(); i++)
			{
				total_out += out_rate[i];
			}
			total_error = total_error / error_rate.size();
			total_out = total_out / out_rate.size();
			
			cout << "avg error is: " << total_error << endl;
			cout << "avg outrate is: " << total_out << endl;
			
#endif