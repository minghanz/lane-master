#include "KeyPts.h"
#include "LaneImage.hpp"

// KeyPts::KeyPts()
// {
//     predicting_left = false;
//     predicting_right = false;
//     renew_flag = false;
// }
void KeyPts::renew(LaneImage& lane_find_image)
{
    if ( lane_find_image.__left_fit!= Vec3f(0, 0, 0) && lane_find_image.__right_fit != Vec3f(0, 0, 0) )
	{
        if (renew_flag)
        {
            markings_left_last = markings_left;
            markings_right_last = markings_right;
        }
        markings_left.clear();
        markings_right.clear();
        
        Mat lane_window_out_r(warp_row, warp_col, CV_8UC1);
        Mat lane_window_out_l(warp_row, warp_col, CV_8UC1);
        Mat out[] = {lane_window_out_r, lane_window_out_l};
        int from_to[] = {2, 0, 0, 1};
        mixChannels(&lane_find_image.__lane_window_out_img, 1, out, 2, from_to, 2);

        lane_out_img_copy = Mat(warp_row + 50, warp_col, lane_find_image.__lane_window_out_img.type(), Scalar(0));
        lane_find_image.__lane_window_out_img.copyTo(lane_out_img_copy.rowRange(0, warp_row));

        cout << "start finding keys " << endl;
		// bool ini_p_left, end_p_left;
		selectPt(lane_window_out_l, lane_out_img_copy, lane_find_image.__plot_pts_lr_warp[0], markings_left);
		cout << "finish key custom 1 " << endl;
        selectPt(lane_window_out_r, lane_out_img_copy, lane_find_image.__plot_pts_lr_warp[1], markings_right);
        cout << "finish key custom 2 " << endl;

        renew_flag = true;


        imshow("key_custom", lane_out_img_copy);
    }
    else
    {
        renew_flag = false;
        predicting_left = false;
        predicting_right = false;
    }

}

void KeyPts::match(LaneImage& lane_find_image)
{
    cout << "entering match" << endl;
    bool left_match_new, right_match_new;
    // int left_match_num, right_match_num;
    left_match_new = markings_left.size() < markings_left_last.size();
    right_match_new = markings_right.size() < markings_right_last.size();
    
    match_side(markings_left, markings_left_last, left_match_new, coord_left, coord_left_last, match_to_idx_left, match_or_not_left );
    match_side(markings_right, markings_right_last, right_match_new, coord_right, coord_right_last, match_to_idx_right, match_or_not_right );
    
    cout << "match_or_not_left " << match_or_not_left << endl;
    cout << "match_or_not_right " << match_or_not_right << endl;
    Mat coord_proj_left, coord_proj_right;
    if (match_or_not_left)
    {
        match_esti(coord_left, coord_left_last, R_left, t_left);
        cout << "coord_left_last.size(): " << coord_left_last.size() << endl;
        coord_proj_left = R_left*coord_left_last; // + t_left;
        coord_proj_left.row(0) = coord_proj_left.row(0) + t_left.at<float>(0);
        coord_proj_left.row(1) = coord_proj_left.row(1) + t_left.at<float>(1);
        
        for (int i = 0; i < coord_proj_left.cols; i++)
        {
            // circle(lane_out_img_copy, Point(coord_left_last.at<float>(0, i),coord_left_last.at<float>(1, i)) , 3, Scalar(0, 255, 255), 1);
            circle(lane_out_img_copy, Point(coord_left.at<float>(0, i),coord_left.at<float>(1, i)) , 3, Scalar(255, 255, 0), 1);
            circle(lane_out_img_copy, Point(coord_proj_left.at<float>(0, i),coord_proj_left.at<float>(1, i)) , 3, Scalar(255, 255, 255), 1);
        }
        compli_last(markings_left, markings_left_last, left_match_new, match_to_idx_left, R_left, t_left, lane_out_img_copy, predicting_left);
        revise_fit(lane_find_image.__left_fit, markings_left, lane_find_image.__lane_window_out_img);
    }
    if (match_or_not_right)
    {
        match_esti(coord_right, coord_right_last, R_right, t_right);

        // cout << "coord_right.size(): " << coord_right.size() << endl;
        // cout << "R_right.size(): " << R_right.size() << endl;
        // cout << "t_right.size(): " << t_right.size() << endl;
        // cout << "coord_right.type(): " << coord_right.type() << endl;
        // cout << "R_right.type(): " << R_right.type() << endl;
        // cout << "t_right.type(): " << t_right.type() << endl;
        
        coord_proj_right = R_right*coord_right_last; // + t_right;
        coord_proj_right.row(0) = coord_proj_right.row(0) + t_right.at<float>(0);
        coord_proj_right.row(1) = coord_proj_right.row(1) + t_right.at<float>(1);
        
        for (int i = 0; i < coord_proj_right.cols; i++)
        {
            // circle(lane_out_img_copy, Point(coord_right_last.at<float>(0, i),coord_right_last.at<float>(1, i)) , 3, Scalar(0, 255, 255), 1);
            circle(lane_out_img_copy, Point(coord_right.at<float>(0, i),coord_right.at<float>(1, i)) , 3, Scalar(255, 255, 0), 1);
            circle(lane_out_img_copy, Point(coord_proj_right.at<float>(0, i),coord_proj_right.at<float>(1, i)) , 3, Scalar(255, 255, 255), 1);
            
        }
        compli_last(markings_right, markings_right_last, right_match_new, match_to_idx_right, R_right, t_right, lane_out_img_copy, predicting_right);
        revise_fit(lane_find_image.__right_fit, markings_right, lane_find_image.__lane_window_out_img);
    }


    imshow("key_custom", lane_out_img_copy);


}

void revise_fit(Vec3f& lane_fit, vector<Vec4f>& markings, Mat& lane_window_out_img )
{
    cout << "lane_fit_2: " << lane_fit(2) << "lane_fit_1: " << lane_fit(1) << endl;
    float top_y = markings[0][1] == -1 ? markings[0][3] : markings[0][1];
    float diff_y = max(markings.back()[3], markings.back()[1]) - top_y;
    if (abs(lane_fit(2)*warp_row*warp_row + lane_fit(1)*warp_row) >= 30 &&  diff_y > warp_row/2 )
    {
        int point_num = markings.size() * 2;
        Mat X_lane(1, point_num, CV_32F);
        Mat Y_lane(3, point_num, CV_32F, Scalar_<float>(1));

        float *line_Y1 = Y_lane.ptr<float>(1);
        float *line_Y2 = Y_lane.ptr<float>(2);
        float *line_X = X_lane.ptr<float>();

        int point_cur = 0;
        for (int i = 0; i < markings.size(); i++)
        {
            if (markings[i][1] != -1)
            {
                line_Y1[point_cur] = warp_row - markings[i][1] - 1;
                line_Y2[point_cur] = (warp_row - markings[i][1] - 1) * (warp_row - markings[i][1] - 1);
                line_X[point_cur] = markings[i][0];
                point_cur++;
            }
            if (markings[i][3] != -1)
            {
                line_Y1[point_cur] = warp_row - markings[i][3] - 1;
                line_Y2[point_cur] = (warp_row - markings[i][3] - 1) * (warp_row - markings[i][3] - 1);
                line_X[point_cur] = markings[i][2];
                point_cur++;
            }
        }
        Y_lane = Y_lane.t();
        Y_lane.resize(point_cur);
        X_lane = X_lane.t();
        X_lane.resize(point_cur);

        Mat lane_fit_mat = Y_lane.inv(DECOMP_SVD) * X_lane;
        Vec3f lane_fit_new = lane_fit_mat.rowRange(0, 3);
        float cur_weight = 0.5;
        lane_fit(0) = lane_fit(0)*(1-cur_weight) + lane_fit_new(0)*cur_weight;
        lane_fit(1) = lane_fit(1)*(1-cur_weight) + lane_fit_new(1)*cur_weight;
        lane_fit(2) = lane_fit(2)*(1-cur_weight) + lane_fit_new(2)*cur_weight;
        

        vector<Point> draw_pts;
        for (int i = 0; i < warp_row; i++)
        {
            int x_cur = lane_fit(0) + lane_fit(1) * i + lane_fit(2) * i * i;
            draw_pts.push_back(Point(x_cur, warp_row - 1 - i));
        }
        vector<vector<Point>> draw_pts_container;
        draw_pts_container.push_back(draw_pts);
        polylines(lane_window_out_img, draw_pts_container, false, Scalar(255, 255, 255), 1, LINE_4);
    }
}

void compli_last(vector<Vec4f>& markings, vector<Vec4f>& markings_last, bool match_from_new, vector<int>& match_to_idx, Mat& R, Mat& t, Mat& lane_out_img_copy, bool& predicting)
{
    int bot = markings.size()-1;// match_num-1;

    if (markings[bot][3] == -1 && markings[bot][1] != -1)   // the last marking shows half
    {
        int last_idx = -1;
        if (match_from_new)
        {
            last_idx = match_to_idx[bot];
        }
        else
        {
            for (int i = 0; i < match_to_idx.size(); i++)
            {
                if (match_to_idx[i] == bot)
                {
                    last_idx = i;
                    break;
                }
            }
        }
        if (last_idx != -1)
        {
            if (markings_last[last_idx][3] != -1 && markings_last[last_idx][1] != -1)
            {
                Matx21f bot_last(markings_last[last_idx][2], markings_last[last_idx][3]);
                Mat bot_last_mat(bot_last);
                Mat last_point = R * bot_last_mat + t;
                markings[bot][2] = last_point.at<float>(0);
                markings[bot][3] = last_point.at<float>(1);
                cout << "bottom point 2n predicted " << endl;
                predicting = true;
                // circle(lane_out_img_copy, Point(markings_last[last_idx][2],markings_last[last_idx][3] ), 3, Scalar(0, 255, 255), -1);
                circle(lane_out_img_copy, Point(markings[bot][2],markings[bot][3]) , 3, Scalar(255, 255, 255), -1);

            }
        }
    }
    else if (predicting)        // the last marking goes out of view
    {
        Vec4f predicted_bot(-1, -1, -1, -1);
        int last_idx = markings_last.size()-1;
        if (last_idx > -1)
        {
            if (markings_last[last_idx][3] != -1 && markings_last[last_idx][1] != -1)
            {
                Matx21f bot_last(markings_last[last_idx][2], markings_last[last_idx][3]);
                Mat bot_last_mat(bot_last);
                Mat last_point = R * bot_last_mat + t;
                predicted_bot[2] = last_point.at<float>(0);
                predicted_bot[3] = last_point.at<float>(1);

                Matx21f bot_last_2(markings_last[last_idx][0], markings_last[last_idx][1]);
                Mat bot_last_mat_2(bot_last_2);
                Mat last_point_2 = R * bot_last_mat_2 + t;
                predicted_bot[0] = last_point_2.at<float>(0);
                predicted_bot[1] = last_point_2.at<float>(1);
                markings.push_back(predicted_bot);
                cout << "bottom point 2n predicted " << endl;
                predicting = true;

                // circle(lane_out_img_copy, Point(markings_last[last_idx][0],markings_last[last_idx][1] ), 3, Scalar(0, 255, 255), -1);
                // circle(lane_out_img_copy, Point(markings_last[last_idx][2],markings_last[last_idx][3] ), 3, Scalar(0, 255, 255), -1);
                circle(lane_out_img_copy, Point(predicted_bot[0],predicted_bot[1]) , 3, Scalar(255, 255, 255), -1);
                circle(lane_out_img_copy, Point(predicted_bot[2],predicted_bot[3]) , 3, Scalar(255, 255, 255), -1);
            }
        }
    }
    
    
    
}


void selectPt(Mat& lane_window_side, Mat& lane_out_img_copy, vector<Point>& plot_pts_warp, vector<Vec4f>& markings)
{
    int up_thresh = 4, low_thresh = 1, dist_thresh = 10;
	vector<int> near_nonznum(plot_pts_warp.size(), 0);

    int start_y = 0; // 100

	if (plot_pts_warp[start_y].x >= 0 && plot_pts_warp[start_y].x < warp_col )
		near_nonznum[start_y] = countNonZero(lane_window_side(Range(plot_pts_warp[start_y].y, plot_pts_warp[start_y].y + 1), Range(max(plot_pts_warp[start_y].x - 5, 0), min(plot_pts_warp[start_y].x + 5, warp_col))));

	cout << "near_nonznum init" << endl;
	
	Point cache_point(0, 0);
    Vec4f cur_marking(-1, -1, -1, -1);
	float cache_ctnonz = 0;
	bool cur_p = near_nonznum[start_y] >= up_thresh;
	// ini_p = cur_p;	// to indicate whether the first keypt is 2n (ini_p = true) or 2p (ini_p = false)
	for (int i = start_y+1; i < plot_pts_warp.size(); i++)
	{
		if (plot_pts_warp[i].x >= 0 && plot_pts_warp[i].x < warp_col )
			near_nonznum[i] = countNonZero(lane_window_side(Range(plot_pts_warp[i].y, plot_pts_warp[i].y + 1), Range(max(plot_pts_warp[i].x - 5, 0), min(plot_pts_warp[i].x + 5, warp_col))));

		cache_ctnonz += near_nonznum[i];

		if ( !cur_p && near_nonznum[i] >= up_thresh)
		{
			bool dist_ok = true;
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
                        cur_marking[2] = cache_point.x;
                        cur_marking[3] = cache_point.y;
                        markings.push_back(cur_marking);
						circle(lane_out_img_copy, cache_point, 3, Scalar(0, 255, 0), -1);
						// key_2n.push_back(cache_point);
                        cur_marking = Vec4f(-1,-1,-1,-1);
						cache_point = plot_pts_warp[i];
						cache_ctnonz = 0;
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
		}
		else if (cur_p && near_nonznum[i] <= low_thresh) 
		{
			bool dist_ok = true;
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
                        cur_marking[0] = cache_point.x;
                        cur_marking[1] = cache_point.y;
						circle(lane_out_img_copy, cache_point, 3, Scalar(0, 255, 0), 1);
						// key_2p.push_back(cache_point);
						cache_point = plot_pts_warp[i];
						cache_ctnonz = 0;
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
		}
	}
	// end_p = cur_p;
	if (cache_point != Point(0, 0) )
	{
		if ( !cur_p )
		{
            cur_marking[2] = cache_point.x;
            cur_marking[3] = cache_point.y;
            markings.push_back(cur_marking);
            circle(lane_out_img_copy, cache_point, 3, Scalar(0, 255, 0), -1);
            cur_marking = Vec4f(-1,-1,-1,-1);
            // key_2n.push_back(cache_point);
			// circle(lane_out_img_copy, key_2n.back(), 3, Scalar(0, 255, 0), -1);
		}
		else
		{
            cur_marking[0] = cache_point.x;
            cur_marking[1] = cache_point.y;
            markings.push_back(cur_marking);
            circle(lane_out_img_copy, cache_point, 3, Scalar(0, 255, 0), 1);
            // key_2p.push_back(cache_point);
			// circle(lane_out_img_copy, key_2p.back(), 3, Scalar(0, 255, 0), 1);
		}
	}
}


void match_esti(Mat& coord, Mat& coord_last, Mat& R, Mat& t)
{
    Scalar_<float> mu_x = mean(coord.row(0));
    Scalar_<float> mu_y = mean(coord.row(1));
    Scalar_<float> mu_x_last = mean(coord_last.row(0));
    Scalar_<float> mu_y_last = mean(coord_last.row(1));
    Mat coord_cp(coord.size(), coord.type());
    Mat coord_last_cp(coord.size(), coord.type());
    coord_cp.row(0) = coord.row(0) - mu_x;
    coord_cp.row(1) = coord.row(1) - mu_y;
    coord_last_cp.row(0) = coord_last.row(0) - mu_x_last;
    coord_last_cp.row(1) = coord_last.row(1) - mu_y_last;


    SVD svd(coord_cp * coord_last_cp.t() );
    R = svd.u * svd.vt;
    
    Matx21f mu(mu_x[0], mu_y[0]);
    Matx21f mu_last(mu_x_last[0], mu_y_last[0]);
    Mat mu_mat(mu);
    Mat mu_last_mat(mu_last);
    
    t = mu_mat - R*mu_last_mat;
    cout << R << endl;
    cout << t << endl;
    
}

void match_side(vector<Vec4f>& markings, vector<Vec4f>& markings_last, bool match_from_new, Mat& coord_new, Mat& coord_last, vector<int>& match_to_idx, bool& match_or_not )
{
    vector<Vec4f>& match_from = match_from_new ? markings : markings_last;
    vector<Vec4f>& match_to = match_from_new ? markings_last : markings;

    
    int match_num = match_from.size();
    match_or_not = false;

    cout << "match_from_new: " << match_from_new << endl;
    cout << "match_num: " << match_num << endl;


    if (match_num >= 2)
    {
        match_to_idx = vector<int>(match_num, -1);
        for (int i = match_from.size()- 1 ; i >= 0; i--)
        {
            float cur_min_dis = 1000300;
            for(int j = match_to.size()-1; j >= 0; j--)
            {
                if (i <= match_from.size()- 2)
                {
                    if (match_to_idx[i+1] != -1 && j >= match_to_idx[i+1])
                        continue;
                }
                float dis_sim = match_eval(match_from[i], match_to[j], match_from_new);
                cout << "dis_sim-" << i << ": " << dis_sim << endl;
                if (dis_sim < cur_min_dis)
                {
                    cur_min_dis = dis_sim;
                    match_to_idx[i] = j;
                }
                else if (dis_sim < 1000000)
                {
                    break;
                }
            }
        }
        cout << "match_to_idx: " ;
        for (int i = 0; i < match_to_idx.size(); i++)
        {
            cout << match_to_idx[i] << " ";
        }
        cout << endl;

        coord_new = Mat(2*match_num, 2, CV_32FC1, Scalar(0));
        coord_last = Mat(2*match_num, 2, CV_32FC1, Scalar(0));
        int cur_row = 0;
        if (match_from_new)
        {
            for (int i = 0; i < match_num; i++)
            {
                if (match_to_idx[i] != -1)
                {
                    if (markings[i][1] != -1 && markings_last[ match_to_idx[i] ][1] != -1)
                    {
                        coord_new.at<float>(cur_row, 0) = markings[i][0];
                        coord_new.at<float>(cur_row, 1) = markings[i][1];
                        coord_last.at<float>(cur_row, 0) = markings_last[ match_to_idx[i] ][0];
                        coord_last.at<float>(cur_row, 1) = markings_last[ match_to_idx[i] ][1];
                        cur_row++;
                    }
                    if (markings[i][3] != -1 && markings_last[ match_to_idx[i] ][3] != -1)
                    {
                        coord_new.at<float>(cur_row, 0) = markings[i][2];
                        coord_new.at<float>(cur_row, 1) = markings[i][3];
                        coord_last.at<float>(cur_row, 0) = markings_last[ match_to_idx[i] ][2];
                        coord_last.at<float>(cur_row, 1) = markings_last[ match_to_idx[i] ][3];
                        cur_row++;
                    }
                    
                }
            }
        }
        else
        {
            for (int i = 0; i < match_num; i++)
            {
                if (match_to_idx[i] != -1)
                {
                    if (markings[ match_to_idx[i] ][1] != -1 && markings_last[i][1] != -1)
                    {
                        coord_new.at<float>(cur_row, 0) = markings[ match_to_idx[i] ][0];
                        coord_new.at<float>(cur_row, 1) = markings[ match_to_idx[i] ][1];
                        coord_last.at<float>(cur_row, 0) = markings_last[i][0];
                        coord_last.at<float>(cur_row, 1) = markings_last[i][1];
                        cur_row++;
                    }
                    if (markings[ match_to_idx[i] ][3] != -1 && markings_last[i][3] != -1)
                    {
                        coord_new.at<float>(cur_row, 0) = markings[ match_to_idx[i] ][2];
                        coord_new.at<float>(cur_row, 1) = markings[ match_to_idx[i] ][3];
                        coord_last.at<float>(cur_row, 0) = markings_last[i][2];
                        coord_last.at<float>(cur_row, 1) = markings_last[i][3];
                        cur_row++;
                    }
                    
                }
            }
        }
        coord_new.resize(cur_row);
        coord_last.resize(cur_row);
        coord_new = coord_new.t();
        coord_last = coord_last.t();
        cout << "cur_row: " << cur_row << endl;
        if (cur_row >= 3)
        {
            match_or_not = true;
        }
    }
}

float match_eval(Vec4f& from, Vec4f& to, bool match_from_new)
{
    Vec4f vec_new = match_from_new ? from : to;
    Vec4f vec_old = match_from_new ? to : from;
    
    int p_1 = (vec_new[1] == -1 || vec_old[1] == -1) ? 0 : 1;
    int p_3 = (vec_new[3] == -1 || vec_old[3] == -1) ? 0 : 1;
    
    float margin = max(10.0, 0.25 * (vec_new[3]- vec_new[1]) ); 
    cout << "new-out: " << vec_new[1] - vec_old[1] << " " << vec_new[3] - vec_old[3] << endl;
    bool wrong_indicator = (vec_new[1] - vec_old[1] + margin )*p_1 <= 0 && (vec_new[3] - vec_old[3] + margin  )*p_3 <= 0 ; // a margin is applied
    
    float disparity = (vec_new[1] - vec_old[1])*(vec_new[1] - vec_old[1])*p_1*(2-p_3) + (vec_new[3] - vec_old[3])*(vec_new[3] - vec_old[3])*p_3*(2-p_1);
    if (wrong_indicator)
        disparity += 1000000;

    return disparity;
}

void KeyPts::renew(vector<Point>& key_left_2p_new, vector<Point>& key_left_2n_new, vector<Point>& key_right_2p_new, 
        vector<Point>& key_right_2n_new, bool& ini_p_left_new, bool& ini_p_right_new, bool& end_p_left_new, bool& end_p_right_new)
{
    key_left_2p = key_left_2p_new;
    key_left_2n = key_left_2n_new;
    key_right_2p = key_right_2p_new;
    key_right_2n = key_right_2n_new;

    ini_p_left = ini_p_left_new;
    ini_p_right = ini_p_right_new;
    end_p_left = end_p_left_new;
    end_p_right = end_p_right_new;
    renew_flag = true;
    
}
// void KeyPts::match()
// {
//     bool left_match_new, right_match_new;
//     int left_match_num, right_match_num;
//     if (key_left_2p.size() < key_left_2p_last.size())
//     {
//         left_match_num = key_left_2p.size();
//         left_match_new = true;
//     }
//     else
//     {
//         left_match_num = key_left_2p_last.size();
//         left_match_new = false;
//     }
//     if (key_right_2p.size() < key_right_2p_last.size())
//     {
//         right_match_num = key_right_2p.size();
//         right_match_new = true;
//     }
//     else
//     {
//         right_match_num = key_right_2p_last.size();
//         right_match_new = false;
//     }
    
    
// }


