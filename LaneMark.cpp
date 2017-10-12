#include "LaneMark.h"
#include "LaneImage.hpp"
#include "Line.hpp"
#include "VanPt.h"

LaneMark::LaneMark()
{
	left_fit_img = Vec3f(0, 0, 0);	// results from LaneImage
    right_fit_img = Vec3f(0, 0, 0);
    left_fit_best = Vec3f(0, 0, 0);	// results from Line's best result
	right_fit_best = Vec3f(0, 0, 0);

	window_half_width = warp_col/12;    // 12 16
	hist_width = 0;

	avg_hist_left_fit = Vec3f(0, 0, 0);       // fed to LaneImage, renewed by recordHistFit
	avg_hist_right_fit = Vec3f(0, 0, 0);
	pos_of_renew_fit_left = 0;          // used by recordHistFit
	pos_of_renew_fit_right = 0;

	new_result = false;
	initial_frame = true;
	last_all_white = false;
}

void LaneMark::recordImgFit(LaneImage& lane_find_image)
{
    left_fit_img = lane_find_image.get_lane_fit(1);
	right_fit_img = lane_find_image.get_lane_fit(2);
	
	initial_frame = lane_find_image.__initial_frame;
	new_result = (left_fit_img != Vec3f(0, 0, 0) && right_fit_img != Vec3f(0, 0, 0));
}

void LaneMark::recordBestFit(Line& left_lane, Line& right_lane)
{
	if ( left_lane.detected && right_lane.detected )
	{
		left_fit_best = left_lane.best_fit;
		right_fit_best = right_lane.best_fit;
		left_fit_w = left_lane.__w_current;
		right_fit_w = right_lane.__w_current;
	
		float mean_dist = getLaneWidthWarp(left_fit_best, right_fit_best);
		window_half_width = (int)(mean_dist/6.0); // 6 / 0.125
	}
	else
	{
		new_result = false;
	}
    
}

void LaneMark::recordHistFit()
{
	if (new_result)
	{
		recordHistFit_(hist_fit_left, avg_hist_left_fit, left_fit_best, pos_of_renew_fit_left, initial_frame);
		recordHistFit_(hist_fit_right, avg_hist_right_fit, right_fit_best, pos_of_renew_fit_right, initial_frame);
		if (avg_hist_left_fit != Vec3f(0, 0, 0) && avg_hist_right_fit != Vec3f(0, 0, 0))
		{
			hist_width = abs(avg_hist_right_fit[0] - avg_hist_left_fit[0]);
		}
	}
    
}

void recordHistFit_(vector<Vec3f>& hist_fit, Vec3f& avg_hist_fit, Vec3f& new_fit, int& pos_of_renew_fit, bool initial_frame) // have delay of three
{
	int size_hist = 20;
	int delay = 3;
	if (hist_fit.empty() )
	{
		hist_fit.push_back(new_fit);
		avg_hist_fit = new_fit;
	}
	else if (initial_frame)
	{
		hist_fit.clear();
		hist_fit.push_back(new_fit);
		pos_of_renew_fit = 0;
		avg_hist_fit = new_fit;
	}
	else if (hist_fit.size() < size_hist + delay)
	{
		if (hist_fit.back() != new_fit)
		{
			hist_fit.push_back(new_fit);
			// if (hist_fit.size() == size_hist + delay)
			// {
			// 	for (int i = 0; i < size_hist; i++)
			// 	{
			// 		avg_hist_fit += hist_fit[i];
			// 	}
			// 	avg_hist_fit = avg_hist_fit*(1.0/(float)size_hist);
			// }
			avg_hist_fit = Vec3f(0,0,0);
			if (hist_fit.size() <= size_hist)
			{
				for (int i = 0; i < hist_fit.size() ; i++)
				{
					avg_hist_fit += hist_fit[i];
				}
				avg_hist_fit = avg_hist_fit*(1.0/(float)hist_fit.size() );
			}
			else
			{
				for (int i = 0; i < size_hist; i++)
				{
					avg_hist_fit += hist_fit[i];
				}
				avg_hist_fit = avg_hist_fit*(1.0/(float)size_hist);
			}
		}
	}
	else if (hist_fit.size() == size_hist + delay)
	{
		if (hist_fit[(pos_of_renew_fit + (size_hist + delay - 1)) % (size_hist + delay)] != new_fit)
		{
			avg_hist_fit = (size_hist*avg_hist_fit - hist_fit[pos_of_renew_fit] + hist_fit[(pos_of_renew_fit + size_hist) % (size_hist + delay)])*(1.0/(float)size_hist);
			hist_fit[pos_of_renew_fit] = new_fit;
			pos_of_renew_fit = (pos_of_renew_fit + 1) % (size_hist + delay);
		}
	}
	else
	{
		cout << "Wrong number of elements in hist_fit." << endl;
		while(hist_fit.size() > size_hist + delay)
		{
			hist_fit.pop_back();
		}
		cout << "Now the length of hist_fit: " << hist_fit.size() << endl;
	}
}

float getLaneWidthWarp(Vec3f left_fit, Vec3f right_fit)
{
	/// average lane width in warped image
	float y_eval_loc = 0; // changed to counting from closer side
	valarray<float> y_eval(y_eval_loc, 10);
	float step = warp_row/10;
	for (int i = 1; i<10; i++)
		y_eval[i] = y_eval[i-1] + step; // changed to counting from closer side
	valarray<float> x_l(10);
	valarray<float> x_r(10);
	x_l = left_fit[2]*y_eval*y_eval + left_fit[1]*y_eval + left_fit[0];
	x_r = right_fit[2]*y_eval*y_eval + right_fit[1]*y_eval + right_fit[0];
	valarray<float> x_dist = abs(x_r-x_l);
	float mean_dist = x_dist.sum()/x_dist.size();
	
	cout << "Lane mean dist: " << mean_dist << endl;
	return mean_dist;
}


void LaneMark::drawOn(Mat& newwarp, vector<Point>& plot_pts_l, vector<Point>& plot_pts_r, VanPt& van_pt)
{
	Size warp_size = Size(warp_col, warp_row);
	Mat warp_zero(warp_size, CV_8UC3, Scalar(0, 0, 0));
	vector<Point> plot_pts;
	plot_pts.reserve(2 * warp_row);
	plot_pts_l.reserve(warp_row);
	plot_pts_r.reserve(warp_row);
	
	for (int i = 0; i < warp_row; i++)
	{
		Point left;
		left.x = left_fit_best[2]*i*i + left_fit_best[1]*i + left_fit_best[0];
		left.y = warp_row - 1 - i; // from downside
		if (left.x >= 0 && left.y >= 0 && left.x <warp_size.width  && left.y < warp_size.height)
		{
			plot_pts.push_back(left);
			plot_pts_l.push_back(left);
		}
	}
	for (int i = warp_row-1; i >= 0; i--)
	{
		Point right;
		right.x = right_fit_best[2]*i*i + right_fit_best[1]*i + right_fit_best[0];
		right.y = warp_row - 1 - i; // from downside
		if (right.x >= 0 && right.y >= 0 && right.x <warp_size.width  && right.y < warp_size.height)
		{
			plot_pts.push_back(right);
			plot_pts_r.push_back(right);
		}
	}
	
	vector<vector<Point> > plot_pts_vec;
	plot_pts_vec.push_back(plot_pts);
	if (left_fit_w * right_fit_w == 0 ) // || left_fit == Vec3f(0, 0, 0) || right_fit == Vec3f(0, 0, 0)
		fillPoly(warp_zero, plot_pts_vec, Scalar(255, 0, 0) );
	else if ( left_fit_w < 0.1 || right_fit_w < 0.1 )
		fillPoly(warp_zero, plot_pts_vec, Scalar(150, 150, 0) );
	else
		fillPoly(warp_zero, plot_pts_vec, Scalar(0, 255, 0) );
		
	Mat newwarp_lanemark(newwarp.size(), CV_8UC3, Scalar(0,0,0));
	warpPerspective(warp_zero, newwarp_lanemark, van_pt.inv_per_mtx, img_size );
	newwarp = newwarp + newwarp_lanemark;
}