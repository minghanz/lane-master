#include "VehMask.h"

VehDetc::VehDetc(Rect& pos_cur, float IOU_cur, double confi_cur)
{
    pos = pos_cur;
    IOU = IOU_cur;
    confi = confi_cur;
    conti_count = 1;
    count = 1;
    conti_valid = false;
    renewed = false;

    lat_speed = 0;
}

void VehDetc::posrenew(Rect& cur_rect, double confi_cur, Mat& per_mtx, Mat& inv_per_mtx)
{
    estside(cur_rect, per_mtx, inv_per_mtx);

    pos = cur_rect;
    confi = confi_cur;
    conti_count = min(10, conti_count + 1);
    count ++;
    renewed = true;
    if (conti_count >= 5)
    {
        conti_valid = true;
    }
}

void VehDetc::negrenew()
{
    conti_count -= 2;
    count ++;
    if (conti_count <= 0)
    {
        conti_valid = false;
    }
}

void VehDetc::estside(Rect& cur_rect, Mat& per_mtx, Mat& inv_per_mtx)
{
    float vert_length = 50;

    float old_x = pos.x + pos.width/2;
    float new_x = cur_rect.x + cur_rect.width/2;
    Vec2f old_left(pos.x, pos.y + pos.height);
    Vec2f old_right(pos.x + pos.width, pos.y + pos.height);
    Vec2f new_left(cur_rect.x, cur_rect.y + cur_rect.height);
    Vec2f new_right(cur_rect.x + cur_rect.width, cur_rect.y + cur_rect.height);
    vector<Vec2f> coord;
    coord.push_back(old_left);
    coord.push_back(old_right);
    coord.push_back(new_left);
    coord.push_back(new_right);
    
    vector<Vec2f> bird_coord;
    perspectiveTransform(coord, bird_coord, per_mtx);
    lat_speed = 0.5 * (lat_speed + ( bird_coord[2][0]+bird_coord[3][0] - (bird_coord[0][0] + bird_coord[1][0]) )/2 ) ;
    // cout << "old_center: " << old_center << endl;
    // cout << "old_center_warp: " << bird_coord[0] << endl;
    
    // cout << "new_center: " << new_center << endl;
    // cout << "new_center_warp: " << bird_coord[1] << endl;
    
    cout << "lat_speed: " << lat_speed << endl;

    float lat_out = lat_speed*2;
    Vec2f front_left(bird_coord[2][0] + lat_out, bird_coord[2][1] - vert_length);
    Vec2f front_right(bird_coord[3][0] + lat_out, bird_coord[3][1] - vert_length);
    vector<Vec2f> bird_front;
    bird_front.push_back(front_left);
    bird_front.push_back(front_right);
    vector<Vec2f> ori_front;
    perspectiveTransform(bird_front, ori_front, inv_per_mtx);
    
    side_poly.clear();
    if (ori_front[0][0] < cur_rect.x)
    {
        vector<Point> left_poly;
        left_poly.push_back(Point(ori_front[0]));
        left_poly.push_back(Point(ori_front[0][0], cur_rect.y));
        left_poly.push_back(Point(cur_rect.x, cur_rect.y));
        left_poly.push_back(Point(cur_rect.x, cur_rect.y + cur_rect.height));       
        side_poly.push_back(left_poly);
    }
    if (ori_front[1][0] > cur_rect.x + cur_rect.width)
    {
        vector<Point> right_poly;
        right_poly.push_back(Point(ori_front[0]));
        right_poly.push_back(Point(ori_front[0][0], cur_rect.y));
        right_poly.push_back(Point(cur_rect.x+cur_rect.width, cur_rect.y));
        right_poly.push_back(Point(cur_rect.x+cur_rect.width, cur_rect.y + cur_rect.height));       
        side_poly.push_back(right_poly);
    }

    
}

VehMask::VehMask() // : tracker(TrackerMIL::create())
{
    hog = HOGDescriptor(Size(50,60), Size(10,10), Size(20,20), Size(10,10), 9, 1 );
    string obj_det_filename = "my_detector.yml";
	hog.load( obj_det_filename );

    // tracker = TrackerKCF::create();
    track_mode = false;

    ori_veh_mask = Mat(img_size, CV_8UC1, Scalar(0));

    #ifndef HIGH_BOT
    sub_img_top = img_size.height*0.5;
    sub_img_bot = img_size.height;
	#else
	sub_img_top = img_size.height*0.3;
    sub_img_bot = img_size.height*0.7;
	#endif

}

void VehMask::detectHOG(Mat& img, Mat& per_mtx, Mat& inv_per_mtx)
{
    Mat subimg;
    img.rowRange(sub_img_top, sub_img_bot).copyTo(subimg);


    ori_veh_mask = Mat(img_size, CV_8UC1, Scalar(0));

    vector<VehDetc> new_track;
    cout << "hog 1" << endl; 
    if ( true) // track_mode == false  // detection (when objects were not found)
    {
        // cur_detc.clear();
        // detection, construct new_track and renew track_detc

        // match new detections with objects in track_detc
        vector<Rect> detections;
        vector<double> foundWeights;

        hog.detectMultiScale(subimg, detections, foundWeights);
        cout << "hog 2" << endl; 
        for (size_t j = 0; j < detections.size(); j++)
        {
            if (foundWeights[j] >= 0.5)
            {
                detections[j].x -= detections[j].width * 0.1;
                detections[j].width *= 1.2;
                Rect cur_rect = detections[j] + Point(0, sub_img_top);

                cout << "hog 3" << endl; 
                if (track_detc.size() == 0)
                {
                    VehDetc cur_wind(cur_rect, 1, foundWeights[j]);
                    new_track.push_back(cur_wind);
                }
                else
                {
                    bool existing = false;
                    for (int i = 0; i < track_detc.size(); i++)
                    {
                        Rect intersection = cur_rect & track_detc[i].pos;
                        if (intersection.area() == 0 || intersection.area() < 0.5 * (cur_rect.area() + track_detc[i].pos.area() - intersection.area()))
                            continue;
                        else
                        {
                            track_detc[i].posrenew(cur_rect, foundWeights[j], per_mtx, inv_per_mtx);
                            new_track.push_back(track_detc[i]);
                            new_track.back().renewed = false;
                            existing = true;
                            break;
                        }
                    }
                    if (existing == false)
                    {
                        VehDetc cur_wind(cur_rect, 1, foundWeights[j]);
                        new_track.push_back(cur_wind);
                    }
                }
            }
        }

        // add qualifying objects in tracking history to new_track
        for (int j = 0; j < track_detc.size(); j++)
        {
            if (track_detc[j].renewed == false)
            {
                track_detc[j].negrenew();
                if (track_detc[j].conti_count > 0)
                {
                    new_track.push_back(track_detc[j]);
                }
            }
        }
        
        // update track_detc using new_track
        track_detc = new_track;

        // find valid_detc from track_detc and initialize trackers
        valid_detc.clear();
        trackers.clear();
        for (int j = 0; j < track_detc.size(); j++)
        {
            if (track_detc[j].conti_valid == true)
            {
                rectangle(ori_veh_mask, track_detc[j].pos, Scalar(255), -1);
                if (track_detc[j].side_poly.size()>0)
                {
                    // polylines(ori_veh_mask, track_detc[j].side_poly, true, Scalar(255), -1);
                    fillPoly(ori_veh_mask, track_detc[j].side_poly, Scalar(255));
                }
                valid_detc.push_back(track_detc[j]);
                track_mode = true;
                cout << "hog 5" << endl; 

                Ptr<Tracker> tracker = TrackerMIL::create();
                Rect sub_coord = track_detc[j].pos - Point(0, sub_img_top);
                Rect2d sub_coord_d;
                sub_coord_d.x = sub_coord.x;
                sub_coord_d.y = sub_coord.y;

                sub_coord_d.width = sub_coord.width;
                sub_coord_d.height = sub_coord.height;
                
                cout << "track: " << track_detc[j].pos.x << " " << track_detc[j].pos.y << " " << track_detc[j].pos.width << " " << track_detc[j].pos.height << " " <<  endl; 
                cout << "modif: " << sub_coord.x << " " << sub_coord.y << " " << sub_coord.width << " " << sub_coord.height << " " <<  endl; 
                // tracker->init( subimg, sub_coord_d );
                // trackers.push_back(tracker);    
                // trackers temporally unused because of initialization problem when running Night_Snake.avi
            }
        }
        cout << "Detection mode." << endl;
    }
    else // tracking (when objects were found), renew valid_detc using new_track, and clear track_detc
    {
        track_detc.clear();
        vector<Ptr<Tracker> > new_trackers;
        cout << "hog 2.1" << endl; 
        for (int i = 0; i < trackers.size(); i++)
        {
            Rect2d cur_track_rect;
            bool track_success = trackers[i]->update(subimg, cur_track_rect);
            if (track_success == true) //  && cur_track_rect.x+cur_track_rect.width < 0.95*img_size.width       // object is still in the scene
            {
                Rect cur_track_rect_int = cur_track_rect;
                if (valid_detc[i].count%5 == 4)         // check whether it is actually an object every 5 frames
                {

                    cout << "Checking mode for " << i << endl;
                    Rect cur_track_rect_large = cur_track_rect_int + Size(cur_track_rect_int.width, cur_track_rect_int.height) - Point(cur_track_rect_int.width*0.5, cur_track_rect_int.height*0.5);
                    if (cur_track_rect_large.x < 0)
                    {
                        cur_track_rect_large.width = min(cur_track_rect_large.x + cur_track_rect_large.width, img_size.width);
                        cur_track_rect_large.x = 0;
                    }
                    else
                    {
                        cur_track_rect_large.width = min(cur_track_rect_large.width, img_size.width - cur_track_rect_large.x);
                    }
                    if (cur_track_rect_large.y < 0)
                    {
                        cur_track_rect_large.height = min(cur_track_rect_large.y + cur_track_rect_large.height, img_size.height/2);
                        cur_track_rect_large.y = 0;
                    }
                    else
                    {
                        cur_track_rect_large.height = min(cur_track_rect_large.height, img_size.height/2 - cur_track_rect_large.y);
                    }
                    Mat near_subimg(subimg, cur_track_rect_large);

                    vector<Rect> detections;
                    vector<double> foundWeights;
                    hog.detectMultiScale(near_subimg, detections, foundWeights);
                    if (detections.size() > 0)                     
                    {
                        double max_idx = 0;
                        for (int j = 0; j < foundWeights.size(); j++)
                        {
                            if (foundWeights[j] > foundWeights[max_idx])
                            {
                                max_idx = j;
                            }
                        }
                        if (foundWeights[max_idx] >= 0.5)   // check passed
                        {
                            // cur_track_rect_int = detections[max_idx] + Point(cur_track_rect_large.x, cur_track_rect_large.y) + Point(0, img_size.height / 2);
                            // rectangle(ori_veh_mask, cur_track_rect_int, Scalar(255), -1);

                            // if(cur_track_rect_int.width * cur_track_rect_int.height == detections[max_idx].width * detections[max_idx].height)
                            // cout << "++++++++++++++++++++++++++++++" << endl;
                            // else
                            // cout << "------------------------------" << endl;
                            
                            // valid_detc[i].posrenew(cur_track_rect_int, foundWeights[max_idx]);
                            // new_track.push_back(valid_detc[i]);
                            // new_track.back().renewed = false;
                            // trackers[i]->init(subimg, cur_track_rect_int-Point(0, img_size.height / 2));
                            // new_trackers.push_back(trackers[i]);

                            cur_track_rect_int = cur_track_rect_int + Point(0, sub_img_top);
                            rectangle(ori_veh_mask, cur_track_rect_int, Scalar(255), -1);
                            valid_detc[i].posrenew(cur_track_rect_int, 1, per_mtx, inv_per_mtx);
                            new_track.push_back(valid_detc[i]);
                            new_track.back().renewed = false;
                            
                            new_trackers.push_back(trackers[i]);
                        }
                    }
                }
                else
                {
                    cout << "Tracking mode for " << i << endl;
                    cur_track_rect_int = cur_track_rect_int + Point(0, sub_img_top);
                    rectangle(ori_veh_mask, cur_track_rect_int, Scalar(255), -1);
                    valid_detc[i].posrenew(cur_track_rect_int, 1, per_mtx, inv_per_mtx);
                    new_track.push_back(valid_detc[i]);
                    new_track.back().renewed = false;

                    new_trackers.push_back(trackers[i]);
                }
            }
        }
        valid_detc = new_track;
        trackers = new_trackers;
        cout << "hog 4.1" << endl; 
        if (trackers.size() == 0)
        {
            track_mode = false;
        }
    }
    

    warpPerspective(ori_veh_mask, warp_veh_mask, per_mtx, Size(warp_col, warp_row));
}

void VehMask::drawOn(Mat& newwarp)
{
    if (valid_detc.size() >= 1)
    {
        vector<Mat> channels;
        split(newwarp, channels);
        channels[2] = ori_veh_mask + channels[2] + 0;
        merge(channels, newwarp);
    }
}

void VehMask::indicateOnWarp(Mat& warped_raw_img)
{
    Mat warp_raw_copy;
    warped_raw_img.copyTo(warp_raw_copy);
    if (valid_detc.size() >= 1)
    {
        vector<Mat> channels;
        split(warp_raw_copy, channels);
        channels[2] = warp_veh_mask + channels[2] + 0;
        merge(channels, warp_raw_copy);
    }
    imshow("warp_raw_copy", warp_raw_copy);
}