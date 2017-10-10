#include "Line.hpp"
#include "LaneImage.hpp"
#include "VanPt.h"
#include "LaneMark.h"

using namespace std;
using namespace cv;

//float ym_per_pix = 40./720.;
//float xm_per_pix = 3.7/600.;

Line::Line(float w_history, bool sanity_check, bool sanity_parallel_check)
{
	detected = false; // get false when current frame output no result or the result has been bad for consequtive 5 frames
	best_fit = Vec3f(0, 0, 0); // Vec and Mat cannot convert to each other
	best_radius_of_curvature = 0;
	best_line_base_pos = 0;
	diffs = Vec3f(0, 0, 0);
	pos_of_renew_diff = 0;
	current_diff = 0;
	mean_hist_diff = 0;
	history_diff.clear();
	this->w_history = w_history;
	check = sanity_check;
	parallel_check = sanity_parallel_check;
	
	best_fit_2 = Vec2f(0, 0); // for vanishing point
}

void Line::processNewRecord(VanPt& van_pt, LaneMark& lane_mark)
{
	bool van_consist = true; // van_pt.consist;
	int window_half_width = lane_mark.window_half_width;
	float w_current = 1 - w_history;
	base_fluctuation = 1.3 * window_half_width;
	if (current_fit.size() >= 1)
	{
		if (current_fit.size() >= 2)
		{
			mean_hist_diff = 0;
			cout << "History_diff: ";
			for (int i = 0; i < history_diff.size(); i++)
			{
				mean_hist_diff += history_diff[i];
				cout << history_diff[i] << " ";
			}
			cout << endl;
			mean_hist_diff = mean_hist_diff / history_diff.size();
			//diffs = current_fit.back() - best_fit;
			//current_diff = abs(diffs[0])+abs(diffs[1])+abs(diffs[2]);
		}
		current_diff = getDiff();
		cout << "Current diff: " << current_diff << ", mean diff: " << mean_hist_diff << ", base_fluctuation: " << base_fluctuation << endl;
		cout << "paralled_check: " << parallel_check << ", check: " << check << endl;
		if ( (current_fit.size() == 1 || detected == false || (mean_hist_diff >= 0.3*base_fluctuation && current_diff < 1.5*base_fluctuation && parallel_check == true))  && van_consist  == true) // the history is bad, not requiring width check, loosing the fluctuation
		{
			if (current_diff < 0.5*base_fluctuation || current_fit.size() == 1 || detected == false)
				w_current = 1;
			else
				w_current = 0.6;
			if (current_fit.size() == 1)
			{
				best_fit = current_fit.back();
				fitRealWorldCurve();
				best_radius_of_curvature = radius_of_curvature.back();
				best_line_base_pos = line_base_pos.back();
				best_fit_2 = current_fit_2.back();
			}
			else
			{
				best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
				fitRealWorldCurve();
				best_radius_of_curvature = getCurvature();
				best_line_base_pos = getDistanceToLane();
				best_fit_2 = w_history*best_fit_2 + w_current*current_fit_2.back();
			}
			//diffs = current_fit.back() - best_fit;
			if (history_diff.size() < 10)
				history_diff.push_back(current_diff);
			else
			{
				history_diff[pos_of_renew_diff] = current_diff;
				pos_of_renew_diff = (pos_of_renew_diff + 1) % 10;
			}
			
			fail_detect_count = 0;
			detected = true;
			cout << "case 1" << endl;
		}
		else if (detected == true && current_diff < 0.5*base_fluctuation && check == true && van_consist  == true) // history is good and current one is good  (modified: check should betrue)
		{
			//w_current = 0.1;
			best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
			fitRealWorldCurve();
			//best_radius_of_curvature = w_history*best_radius_of_curvature + w_current*radius_of_curvature.back();
			//best_line_base_pos = w_history*best_line_base_pos + w_current*line_base_pos.back();
			best_radius_of_curvature = getCurvature();
			best_line_base_pos = getDistanceToLane();
			//diffs = current_fit.back() - best_fit;
			if (history_diff.size() < 10)
				history_diff.push_back(current_diff);
			else
			{
				history_diff[pos_of_renew_diff] = current_diff;
				pos_of_renew_diff = (pos_of_renew_diff + 1) % 10;
			}
			
			best_fit_2 = w_history*best_fit_2 + w_current*current_fit_2.back();
			
			fail_detect_count = 0;
			cout << "case 2" << endl;
			
		}
		else if (detected == true && current_diff < 1*base_fluctuation && check == true && van_consist  == true) // history is soso(have fluctuation) and current one is good (modified: check should betrue)
		{
			w_current *= 0.6;
			best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
			fitRealWorldCurve();
			//best_radius_of_curvature = (1 - w_current)*best_radius_of_curvature + w_current*radius_of_curvature.back();
			//best_line_base_pos = (1 - w_current)*best_line_base_pos + w_current*line_base_pos.back();
			best_radius_of_curvature = getCurvature();
			best_line_base_pos = getDistanceToLane();
			//diffs = current_fit.back() - best_fit;
			if (history_diff.size() < 10)
				history_diff.push_back(current_diff);
			else
			{
				history_diff[pos_of_renew_diff] = current_diff;
				pos_of_renew_diff = (pos_of_renew_diff + 1) % 10;
			}
			
			best_fit_2 = (1 - w_current)*best_fit_2 + w_current*current_fit_2.back();
			
			fail_detect_count = 0;
			cout << "case 3" << endl;
		}
		//else if (detected == false && current_diff < 1*base_fluctuation && check == true && bool  == true )// last frame failed but history is good (has excluded case 1)
		//{
			//if ( current_diff <  0.5*base_fluctuation)
				//w_current = 0.5;
			//else
				//w_current = 0.25;
			

			//best_fit = (1 - w_current)*best_fit + w_current*current_fit.back();
			//fitRealWorldCurve();
			////best_radius_of_curvature = (1 - w_current)*best_radius_of_curvature + w_current*radius_of_curvature.back();
			////best_line_base_pos = (1 - w_current)*best_line_base_pos + w_current*line_base_pos.back();
			//best_radius_of_curvature = getCurvature();
			//best_line_base_pos = getDistanceToLane();
			
			//best_fit_2 = (1 - w_current)*best_fit_2 + w_current*current_fit_2.back();
			
			////diffs = current_fit.back() - best_fit;
			//if (history_diff.size() < 10)
				//history_diff.push_back(current_diff);
			//else
			//{
				//history_diff[pos_of_renew_diff] = current_diff;
				//pos_of_renew_diff = (pos_of_renew_diff + 1) % 10;
			//}
			
			//fail_detect_count = 0;
			//detected = true;
			//cout << "case 4" ;
			
		//}
		else // the first 4 cases have emulate all (current_diff < window_half_width && check == true && bool  == true)
		{
			w_current = 0;
			fail_detect_count ++;
			if (fail_detect_count >= 5) detected = false;
			cout << "case 5" << endl;
		}
		__w_current = w_current;
	}
	return;
}

void Line::setSanityCheck(bool sanity_check, bool sanity_parallel_check)
{
	check = sanity_check;
	parallel_check = sanity_parallel_check;
	return;
}

bool Line::getSanityCheck()
{
	return check;
}


float Line::getCurvature()
{
	float y_eval_loc = 0*ym_per_pix; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + 0.1; // changed to counting from closer side
	
	valarray<float> curverad(10);
	curverad = pow(( 1 + pow(( 2*best_fit_cr[2]*y_eval*ym_per_pix +best_fit_cr[1]), 2)),1.5)/abs(2*best_fit_cr[2]);
	float mean_curverad = curverad.sum()/curverad.size();
	return mean_curverad;
}

float Line::getDistanceToLane()
{
	float y_eval_loc =0*ym_per_pix; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + 0.1; // changed to counting from closer side
	
	float veh_loc = warp_col/2*xm_per_pix;
	valarray<float> x0(10);
	x0 = best_fit_cr[2]*y_eval*y_eval + best_fit_cr[1]*y_eval + best_fit_cr[0];
	float distance = x0.sum()/x0.size() - veh_loc;
	
	#ifndef NDEBUG
	cout << "lane-veh loc: " << veh_loc << endl;
	cout << "lane-lane loc: " << x0.sum()/x0.size() << endl;
	#endif
	return distance;
}

void Line::fitRealWorldCurve()
{
	best_fit_cr[0] = best_fit[0]*xm_per_pix;
	best_fit_cr[1] = best_fit[1]*xm_per_pix/ym_per_pix;
	best_fit_cr[2] = best_fit[2]*xm_per_pix/ym_per_pix/ym_per_pix;
	//cout << "real formula from fuse:  " << best_fit_cr << endl;
	return;
}

float Line::getDiff()
{
	/// average lane width in warped image
	float y_eval_loc = 0; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	float step = warp_row/10;
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + step; // changed to counting from closer side
		
	valarray<float> x_new(10);
	x_new = current_fit.back()[2]*y_eval*y_eval + current_fit.back()[1]*y_eval + current_fit.back()[0];
	
	float mean_dist;
	if (current_fit.size() > 1)
	{
		valarray<float> x_samp = best_fit[2]*y_eval*y_eval + best_fit[1]*y_eval + best_fit[0];
		valarray<float> x_dist = abs(x_new-x_samp);
		mean_dist = x_dist.max(); // the maximum is more indicating
		// mean_dist = x_dist.sum()/x_dist.size();
		cout << "Dist: ";
		for (int i = 0; i < 10; i++)
		{
			cout << x_dist[i] << " ";
		}
		cout << endl;
	}
	else
		mean_dist = 0;
	
	cout << "Mean dist between new fit and last best: " << mean_dist << endl;
	return mean_dist;
}


void Line::pushNewRecord(LaneImage& lane_find_image, int direction) // direction = 1: left, 2:right
{
	///lane renew and processing
	float dist2side = lane_find_image.get_distance_to_lane(direction);
	float curvature = lane_find_image.get_curvature(direction);
	Vec3f img_fit = lane_find_image.get_lane_fit(direction);

	current_fit.push_back(img_fit);
	radius_of_curvature.push_back(curvature);
	line_base_pos.push_back(dist2side);
	
	if (direction == 1)
	{
		current_fit_2.push_back(lane_find_image.__left_fit_2_img);
	}
	else
	{
		current_fit_2.push_back(lane_find_image.__right_fit_2_img);
	}
	
	bool sanity_check = lane_find_image.__parallel_check && lane_find_image.__width_check;
	setSanityCheck(sanity_check, lane_find_image.__parallel_check);
}