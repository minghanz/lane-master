#include "LaneImage.hpp"
#include "Line.hpp"
#include "VanPt.h"
#include "LaneMark.h"
#include "LearnModel.h"
#include "VehMask.h"
#include "KeyPts.h"

#include "macro_define.h"

#include <ctime>

using namespace std;
using namespace cv;

Size img_size;

int main(int argc, char** argv)
{
	////////////////////////// Calibration
	bool cali = true; // calibrate using txt file (true) or using default value (false)
	if (argc >= 3)
	{
		string flag_cali = argv[2];
		if (flag_cali == "n")
			cali = false;
	}
	
	/// Camera intrinsic matrix and distortion coefficient
	Mat cam_mtx(3, 3, CV_64FC1, Scalar(0)); 
	Mat dist_coeff;
	/// Horizontal and vertical angle of view
	float alpha_w, alpha_h; 

	/// Read calibration information from txt file
	char cam_mtx_param[200], dist_coeff_param[200], van_param_s[200];
	// ifstream cali_file("../prj_cali/calib_red_mkz_webcam.txt");
	// ifstream cali_file("../prj_cali/calib_red_mkz_webcam_mcity.txt");
	ifstream cali_file("../prj_cali/calib_white_mkz_pointgrey.txt");
	cali_file.getline(cam_mtx_param, 200);
	cali_file.getline(dist_coeff_param, 200);
	cali_file.getline(van_param_s, 200);

	/// Parse the information for vanishing point 
	/// (outside of cali because vanishing point info may be provided without camera intrinsic parameters)
	vector<float> van_param(3, 0);
	sscanf(van_param_s, "%f %f %f", &(van_param[0]), &(van_param[1]), &(van_param[2]));
	cout << "van_param: " << van_param[0] << " " << van_param[1] << " " << van_param[2] << endl;

	if (cam_mtx_param[0] == '0')
	{
		cali = false;
	}
	if (cali)
	{
		//  initialize for camera calibration
		// vector<vector<Point3f> > obj_pts;
		// vector<vector<Point2f> > img_pts;
		// Size image_size;
		// cameraCalibration(obj_pts, img_pts, image_size, 6, 9);
		
		// vector<Mat> rvecs, tvecs;
		// calibrateCamera(obj_pts, img_pts, image_size, cam_mtx, dist_coeff, rvecs, tvecs);

		/// Parse the information for camera intrinsic matrix
		sscanf(cam_mtx_param, "%lf %lf %lf %lf", &cam_mtx.at<double>(0, 0), &cam_mtx.at<double>(1, 1), &cam_mtx.at<double>(0, 2), &cam_mtx.at<double>(1, 2));
		cam_mtx.at<double>(2, 2) = 1;
		
		/// Parse the information for camera distortion coefficients
		vector<double> dist_coeff_vec;
		for (int i = 0, j = 0; i < strlen(dist_coeff_param); i++)
		{
			if (dist_coeff_param[i] == ' ')
			{
				double temp_param;
				string temp(dist_coeff_param + j, i - j);
				stringstream ss;
				ss << temp;
				ss >> temp_param;
				dist_coeff_vec.push_back(temp_param);
				j = i+1;
			}
		}
		dist_coeff = Mat(1, dist_coeff_vec.size(), CV_64FC1);
		for (int i = 0; i < dist_coeff_vec.size(); i++)
		{
			dist_coeff.at<double>(i) = dist_coeff_vec[i];
		}
		
		cout << "cameraMatrix: " << cam_mtx << endl;
		cout << "dist_coeff: " << dist_coeff << endl;


		cout << "size of cam_mtx: " << cam_mtx.size() << endl;
		cout << "depth of cam_mtx: " << cam_mtx.depth() << endl;
		cout << "size of dist_coeff: " << dist_coeff.size() << endl;
		cout << "depth of dist_coeff: " << dist_coeff.depth() << endl;
		getchar();
		cout << cam_mtx.at<double>(0,2) << " " << cam_mtx.at<double>(0,0) << endl;
		cout << cam_mtx.at<double>(1,2) << " " << cam_mtx.at<double>(1,1) << endl;
		
		/// Angle of view (one side)
		alpha_w = atan2(cam_mtx.at<double>(0,2), cam_mtx.at<double>(0,0)); // horizontal
		alpha_h = atan2(cam_mtx.at<double>(1,2), cam_mtx.at<double>(1,1)); // vertical
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
	ofstream pointfile("lane_points.txt");
	
	/// initialize parameters that work cross frames 
	clock_t t_start = clock();
	float time_step = msc[0]; 		// counting frames, fed to LaneImage(__nframe)
	img_size = Size(reader.get(CV_CAP_PROP_FRAME_WIDTH), reader.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout << "img_size: " << img_size << endl;

	Line left_lane, right_lane;
	VanPt van_pt(alpha_w, alpha_h, van_param);
    LaneMark lane_mark;
	LearnModel learn_model;
	VehMask veh_masker;
	KeyPts key_pts;
	// HOGDescriptor hog(Size(50,60), Size(10,10), Size(20,20), Size(10,10), 9, 1 );
    // string obj_det_filename = "my_detector.yml";
	// hog.load( obj_det_filename );


	float illu_comp = 1;
	// int nframe = msc[0];
	
	#ifdef EVA // 1st paragraph
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

			// imshow("image", image);
			// waitKey(0);
			imshow("cali_image", cali_image);
			// waitKey(0);

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
			if (key_pts.renew_flag)
			{
				key_pts.match(lane_find_image);
			}
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

				lane_mark.recordBestFit(left_lane, right_lane, van_pt);
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
				pointfile << time_step << " ";
				lane_mark.drawOn(newwarp, van_pt, lane_find_image, pointfile);

				#ifdef EVA // 2nd paragraph
				#endif
			}
			#ifdef EVA // 3rd paragraph
			#endif
				
			#ifdef DRAW
			/// add results on calibrated image, output the frame
			addWeighted(cali_image, 0.8, newwarp, 0.3, 0, result);
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
				
				string Text1 = "Frame " + to_string((int)time_step);
				// string Text2 = "Hist width: " + x2str(lane_mark.hist_width) + ", min width: " + x2str(lane_find_image.__min_width_warp) + ", width: " + x2str(lane_find_image.__bot_width);
				// string Text3 = "L: cur diff: " + x2str(left_lane.current_diff) + ", mean diff: " + x2str(left_lane.mean_hist_diff) +  ", base_fluc: " + x2str(left_lane.base_fluctuation) + ", w: " + x2str(left_lane.__w_current);
				// string Text4 = "R: cur diff: " + x2str(right_lane.current_diff) + ", mean diff: " + x2str(right_lane.mean_hist_diff) +  ", base_fluc: " + x2str(right_lane.base_fluctuation) + ", w: " + x2str(right_lane.__w_current);
				putText(result, Text1, Point(10, 40), fontFace, fontScale, Scalar(0,0,255), thickness, LINE_AA); //Point(image.cols/10, 40)
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
			#ifdef EVA // 4th paragraph			
			#endif
			cout << "total process time: " << to_string(((float)(clock() - t_start))/CLOCKS_PER_SEC) << "s" << endl;
			break;
		}
	}
	outfile.close();
	pointfile.close();
	return 0;
}
