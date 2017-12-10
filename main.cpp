#include "LaneImage.hpp"
#include "Line.hpp"
#include "VanPt.h"
#include "LaneMark.h"
#include "LearnModel.h"
#include "VehMask.h"
#include "KeyPts.h"

#include "macro_define.h"

#include "alglib/interpolation.h"

#include <ctime>

using namespace std;
using namespace cv;

Size img_size;

int main(int argc, char** argv)
{
	bool cali = true;
	if (argc >= 3)
	{
		string flag_cali = argv[2];
		if (flag_cali == "n")
			cali = false;
	}
	
	Mat cam_mtx, dist_coeff; // camera matrix(inner parameters) and distortion coefficient
	float alpha_w, alpha_h; // horizontal and vertical angle of view
	
	if (cali)
	{
		/// initialize for camera calibration
		vector<vector<Point3f> > obj_pts;
		vector<vector<Point2f> > img_pts;
		Size image_size;
		cameraCalibration(obj_pts, img_pts, image_size, 6, 9);
		
		vector<Mat> rvecs, tvecs;
		calibrateCamera(obj_pts, img_pts, image_size, cam_mtx, dist_coeff, rvecs, tvecs);
		
		cout << "cameraMatrix: " << cam_mtx << endl;
		cout << cam_mtx.at<double>(0,2) << " " << cam_mtx.at<double>(0,0) << endl;
		cout << cam_mtx.at<double>(1,2) << " " << cam_mtx.at<double>(1,1) << endl;
		
		alpha_w = atan2(cam_mtx.at<double>(0,2), cam_mtx.at<double>(0,0)); // angle of view horizontal
		alpha_h = atan2(cam_mtx.at<double>(1,2), cam_mtx.at<double>(1,1)); // angle of view vertical
	}
	else
	{
		alpha_w = 20*CV_PI/180; // default value
		alpha_h = 20*CV_PI/180; //
	}
	
	
	/// import source file
	string file_name;
	Mat image;
	VideoCapture reader;
	VideoWriter writer;
	int msc[2], ispic, video_length;
	import(argc, argv, file_name, image, reader, writer, msc, ispic, video_length);
	cout << "Set time interval: [" << msc[0] <<", "<< msc[1] << "]" << endl;
	ofstream outfile("yaw angle.txt");
	
	/// initialize parameters that work cross frames 
	clock_t t_start = clock();
	float time_step = msc[0]; 		// counting frames, fed to LaneImage(__nframe)
	img_size = Size(reader.get(CV_CAP_PROP_FRAME_WIDTH), reader.get(CV_CAP_PROP_FRAME_HEIGHT));

	Line left_lane, right_lane;
	VanPt van_pt(alpha_w, alpha_h);
    LaneMark lane_mark;
	LearnModel learn_model;
	VehMask veh_masker;
	KeyPts key_pts;
	// HOGDescriptor hog(Size(50,60), Size(10,10), Size(20,20), Size(10,10), 9, 1 );
    // string obj_det_filename = "my_detector.yml";
	// hog.load( obj_det_filename );


	float illu_comp = 1;
	// int nframe = msc[0];
	
	#ifdef EVA
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
	
	while (reader.isOpened())
	{
		reader.read(image);
		if(!image.empty() && (msc[1] == 0 || time_step <= msc[1])) // reader.get(CV_CAP_PROP_POS_MSEC) <= msc[1]
		{
			clock_t t0 = clock(); // for loop time
			clock_t t_last = clock();  // for step time
			
			// cout << "current time(msc): " << reader.get(CV_CAP_PROP_POS_MSEC) << endl;
			
			
			/// initialize for first frame
			Mat cali_image;
			if (cali)
			{
				undistort(image, cali_image, cam_mtx, dist_coeff);
			}
			else
			{
				image.copyTo(cali_image);
			} // not copied

			#ifndef NDEBUG
			// cout << "cali image size" << cali_image.size() << endl;
			// namedWindow("original color", WINDOW_AUTOSIZE);
			// imshow("original color", image);
			// namedWindow("warped color", WINDOW_AUTOSIZE);
			// imshow("warped color", cali_image);
			#endif

			/// get vanishing point, warp source region, illumination compensation.
			Mat gray, warped_img;
			cvtColor(cali_image, gray, COLOR_BGR2GRAY);
			illuComp(cali_image, gray, illu_comp);
			van_pt.initialVan(cali_image, gray, warped_img, lane_mark);

			// image = image*illu_comp;

			clock_t t_now = clock();
			cout << "Image prepared, using: " << to_string(((float)(t_now - t_last)) / CLOCKS_PER_SEC) << "s. " << endl;
			t_last = t_now;


			// Mat subimg;
			// cali_image.rowRange(cali_image.rows / 2, cali_image.rows).copyTo(subimg);

			veh_masker.detectHOG(cali_image, van_pt.per_mtx, van_pt.inv_per_mtx);
			// vector< Rect > detections;
			// vector< double > foundWeights;
			// vector<Rect> veh_rect;
			// // Mat veh_mask(img_size, CV_8UC1, Scalar(0));

			// // Mat subimg;
			// // cali_image.rowRange(cali_image.rows/2, cali_image.rows).copyTo(subimg);
			// hog.detectMultiScale( subimg, detections, foundWeights );
			// for ( size_t j = 0; j < detections.size(); j++ ){
			// 	if (foundWeights[j] >= 0.5){
			// 		detections[j].x -= detections[j].width*0.1;
			// 		detections[j].width *= 1.2;
			// 		veh_rect.push_back(detections[j] + Point(0,img_size.height/2));
			// 		rectangle( veh_mask, detections[j] + Point(0,img_size.height/2), Scalar(255), -1 );
			// 	}
			// }

			// warpPerspective(veh_mask, warp_veh_mask, van_pt.per_mtx, Size(warp_col, warp_row) );
			if (warped_img.rows > 0)
			{
				veh_masker.indicateOnWarp(warped_img);
			}

			LaneImage lane_find_image(warped_img, van_pt, lane_mark, learn_model, veh_masker, key_pts, time_step);

			t_now = clock();
			cout << "Image processed, using: " << to_string(((float)(t_now - t_last)) / CLOCKS_PER_SEC) << "s. " << endl;
			t_last = t_now;

			key_pts.renew(lane_find_image);
			// if (key_pts.renew_flag)
			// {
			// 	key_pts.match(lane_find_image);
			// }
			lane_mark.recordImgFit(lane_find_image);

			Mat result;
			if ( !lane_mark.new_result )
			{
				left_lane.detected = false;
				right_lane.detected = false;
			}
			else
			{
				left_lane.pushNewRecord(lane_find_image, 1);
				right_lane.pushNewRecord(lane_find_image, 2);
				
				// van_pt.recordHistVan( lane_find_image, left_lane,  right_lane);

				left_lane.processNewRecord(van_pt, lane_mark);
				right_lane.processNewRecord(van_pt, lane_mark);

				learn_model.pushNewRecord(lane_find_image);

				lane_mark.recordBestFit(left_lane, right_lane);
				lane_mark.recordHistFit();

				// van_pt.recordBestVan(left_lane, right_lane);
				
				#ifdef COUT
				cout << "window half width: " << lane_mark.window_half_width << endl;
				cout << "best left fit" << lane_mark.left_fit_best << " " << lane_mark.right_fit_best  << endl;
				
				cout << "yaw: " << lane_mark.theta_w << " d; pitch: " << lane_mark.theta_h << " d;" << endl;
				outfile << lane_mark.theta_w << endl;
				#endif
				t_now = clock();
				cout << "Lane renewed, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
				t_last = t_now;
			}
			
			/// draw estimated lane in warped view, then change to original view, draw vanishing point and draw transform source region
			Mat newwarp(image.size(), CV_8UC3, Scalar(0, 0, 0));
			

			van_pt.drawOn( newwarp, lane_mark);
			
			veh_masker.drawOn(newwarp);
			// if (veh_masker.cur_detc.size() >= 1)
			// {
			// 	vector<Mat> channels;
			// 	split(newwarp, channels);
			// 	channels[2] = veh_mask + channels[2] + 0;
			// 	merge(channels, newwarp);
			// }

			if ( lane_mark.new_result ) // current frame succeeds
			{
				vector<Point> plot_pts_l, plot_pts_r;
				lane_mark.drawOn(newwarp, plot_pts_l, plot_pts_r, van_pt, lane_find_image);
			
				#ifdef EVA
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
			}
			#ifdef EVA
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
				
			#ifdef DRAW
			/// add results on calibrated image, output the frame
			addWeighted(cali_image, 1, newwarp, 0.3, 0, result);
			//// circle(result, lane_find_image.__best_van_pt, 5, Scalar(0, 0, 255), -1); // should be the same as Point(last_van_pt)
			Scalar van_color = van_pt.ini_success ? Scalar(0,0,255) : Scalar(0,255,0);
			circle(result, Point(van_pt.van_pt_ini), 3, van_color, -1); // should be the same as Point(last_van_pt)

			/// add the warped_filter_image to the output image
			if (van_pt.ini_flag)
			{
				Mat small_lane_window_out_img;
				int small_height = img_size.height*0.4;
				int small_width = (float)warp_col / (float)warp_row * small_height;
				resize(lane_find_image.__lane_window_out_img, small_lane_window_out_img, Size(small_width, small_height));
				result( Range(0, small_height), Range(img_size.width - small_width, img_size.width) ) = small_lane_window_out_img + 0;
			}

			double fontScale;
			int thickness;
			if (image.cols < 900)
			{
				fontScale = 0.3;
				thickness = 1;
			}
			else
			{
				fontScale = 0.8;
				thickness = 2;
			}
			int fontFace = FONT_HERSHEY_SIMPLEX;
			if ( !lane_mark.new_result )
			{
				string TextL = "Frame " + to_string((int)time_step);
				putText(result, TextL, Point(10, 40), fontFace, fontScale, Scalar(0,0,255), thickness, LINE_AA);
				putText(result, "Current frame failed.", Point(10, 60), fontFace, fontScale, Scalar(0,0,255), thickness, LINE_AA);
				//// if (lane_find_image.__last_left_fit == Vec3f(0, 0, 0) || lane_find_image.__last_right_fit == Vec3f(0, 0, 0))
				if (lane_mark.initial_frame)
				{
					string TextIni = "Initializing frame. ";
					putText(result, TextIni, Point(10, 80), fontFace, fontScale, Scalar(0,0,255), thickness, LINE_AA);
				}
				#ifndef NDEBUG
				imshow("result", result);
				waitKey(0);
				#else
				imshow("result", result);
				waitKey(1);
				#endif
				
				t_now = clock();
				cout << "Current frame failed. "<< endl;
				cout << "Frame constructed, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
				t_last = t_now;
			}
			else
			{
				#ifdef COUT
				cout << "revised vanishing point: " << van_pt.van_pt_ini << endl;
				cout << "lateral offset left: " << left_lane.best_line_base_pos << endl;
				cout << "lateral offset right: " << right_lane.best_line_base_pos << endl;
				#endif
				
				// string Text1 = "Frame " + to_string((int)time_step);
				// string Text2 = "Hist width: " + x2str(lane_mark.hist_width) + ", min width: " + x2str(lane_find_image.__min_width_warp) + ", width: " + x2str(lane_find_image.__bot_width);
				// string Text3 = "L: cur diff: " + x2str(left_lane.current_diff) + ", mean diff: " + x2str(left_lane.mean_hist_diff) +  ", base_fluc: " + x2str(left_lane.base_fluctuation) + ", w: " + x2str(left_lane.__w_current);
				// string Text4 = "R: cur diff: " + x2str(right_lane.current_diff) + ", mean diff: " + x2str(right_lane.mean_hist_diff) +  ", base_fluc: " + x2str(right_lane.base_fluctuation) + ", w: " + x2str(right_lane.__w_current);
				// putText(result, Text1, Point(10, 40), fontFace, fontScale, Scalar(0,0,200), thickness, LINE_AA); //Point(image.cols/10, 40)
				// putText(result, Text2, Point(10, 60), fontFace, fontScale, Scalar(0,0,200), thickness, LINE_AA); //Point(image.cols/10, 40)
				// putText(result, Text3, Point(10, 80), fontFace, fontScale, Scalar(0,0,200), thickness, LINE_AA); //Point(image.cols/10, 40)
				// putText(result, Text4, Point(10, 100), fontFace, fontScale, Scalar(0,0,200), thickness, LINE_AA); //Point(image.cols/10, 40)
				
				
				#ifndef NDEBUG
				imshow("result", result);
				imshow("result2", lane_find_image.__lane_window_out_img);
				waitKey(0);
				#endif
				
				#ifdef NDEBUG
				imshow("result", result);
				imshow("result2", lane_find_image.__lane_window_out_img);
				waitKey(1);
				#endif
				
				t_now = clock();
				cout << "Frame constructed, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
				t_last = t_now;
				
				// van_pt.renewWarp();
				
				// t_now = clock();
				// cout << "Vanishing point renewed, using: " << to_string(((float)(t_now - t_last))/CLOCKS_PER_SEC) << "s. " << endl;
				// t_last = t_now;
			}	
			
			writer.write(result);
			#endif

			time_step += 1;
			cout << "Frame " << time_step <<". Processed " <<  to_string((int)(time_step/video_length*100)) << "%. ";
			cout << "Process time: " << to_string(((float)(clock() - t0))/CLOCKS_PER_SEC) << "s per frame" << endl << endl;
		}
		else
		{
			#ifdef EVA
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
			cout << "total process time: " << to_string(((float)(clock() - t_start))/CLOCKS_PER_SEC) << "s" << endl;
			break;
		}
	}
	outfile.close();
	return 0;
}
