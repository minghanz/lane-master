#include "LaneImage.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

void cameraCalibration(vector<vector<Point3f> >& obj_pts, vector<vector<Point2f> >& img_pts, Size& image_size, int ny, int nx)
{
	vector<Point3f> objp;
	vector<Point2f> imgp;
	
	string first_file = "../usb_cali_images/cali%d.jpg"; // for MKZ USB camera
	// string first_file = "../camera_cal/calibration%d.jpg";
	VideoCapture calib_imgseq(first_file);
	if (!calib_imgseq.isOpened())
	{
	    cout  << "Could not open the calibration image sequence: " << first_file << endl;
	    return;
	}
	
	for(int i = 0; i<ny*nx; i++)
	{
		objp.push_back(Point3f(i%nx, i/nx, 0)); 
		/*
		#ifndef NDEBUG
		cout << "objp: " << objp[i] << endl;
		#endif
		*/
	}
	
	Mat calib_img, gray;
	Size patternsize(nx, ny);
	for (;;)
	{
		calib_imgseq.read(calib_img);
		if (calib_img.empty()) break;
		image_size = calib_img.size();
		cvtColor(calib_img, gray, COLOR_BGR2GRAY);
		bool patternfound = findChessboardCorners(gray, patternsize, imgp);
		
		if (patternfound)
		{
			obj_pts.push_back(objp);
			img_pts.push_back(imgp);
			/*
			#ifndef NDEBUG
			drawChessboardCorners(calib_img, patternsize, imgp, patternfound);
			namedWindow("calib-image", WINDOW_AUTOSIZE);
			imshow("calib-image", calib_img);
			waitKey(0);
			#endif
			*/
		}
	}
	return;
}


void import(int argc, char** argv, string& file_name, Mat& image, VideoCapture& reader, VideoWriter& writer, int msc[], int& ispic, int& video_length)
{
	if (argc>1) file_name = argv[1];
	else file_name = "../challenge.avi";
	
	if (file_name == "camera")
	{
		reader.open(0);              // Open input
	    if (!reader.isOpened())
	    {
	        cout  << "Could not open the input video: " << file_name << endl;
	        return;
	    }
	    cout << "Camera stream opened successfully. " << endl;
	    
	    int codec = static_cast<int>(reader.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
		Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
               (int) reader.get(CAP_PROP_FRAME_HEIGHT));
                  
		const string out_name = file_name + "_res.avi";
		writer.open(out_name, codec, reader.get(CAP_PROP_FPS), S, true);
		cout << codec << " " << S << " " << reader.get(CAP_PROP_FPS) << endl;
	    return;
	}
	
	// check the format of source file
	const string::size_type dotpos = file_name.find_last_of('.');
	const string::size_type slashpos = file_name.find_last_of('/');
	const string extension = file_name.substr(dotpos+1);
	size_t percpos = file_name.find('%');
	if((extension == "jpg" || extension == "png" )&& percpos == string::npos)
	{
		ispic = 1;
		cout << "Format: picture." << endl;
		image = imread(file_name, IMREAD_COLOR);
		if(image.empty())
		{
			cout << "Could not open or find the image." << endl;
			return;
		}
	}
	else if(extension == "avi" || ( (extension == "jpg" ||extension == "png") && percpos != string::npos ) )
	{
		ispic = 0;
		cout << "Format: video." << endl;
		
		reader.open(file_name);              // Open input
	    if (!reader.isOpened())
	    {
	        cout  << "Could not open the input video: " << file_name << endl;
	        return;
	    }
	    
        string out_name;
        if (extension == "avi")
        {
			video_length = (int) reader.get(CAP_PROP_FRAME_COUNT);
			// retrieve the property of source video
			int codec = static_cast<int>(reader.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
			Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) reader.get(CAP_PROP_FRAME_HEIGHT));
                  
			const string out_name = file_name.substr(slashpos+1,dotpos-slashpos-1) + "2_res.avi";
			writer.open(out_name, codec, reader.get(CAP_PROP_FPS), S, true);
			cout << codec << " " << S << " " << reader.get(CAP_PROP_FPS) << endl;
		}
		else
		{
			int codec = 1196444237;
			double fps = 10;
			Size S = Size((int) reader.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) reader.get(CAP_PROP_FRAME_HEIGHT));
            
            const string out_name = file_name.substr(slashpos+1,percpos-slashpos-1) + "_res.avi";
            writer.open(out_name, codec, fps, S, true);
			cout << codec << " " << S << " " << fps << endl;
		}
		// construct the output class
		
        if (!writer.isOpened())
		{
			cout  << "Could not open the output video for write: " << out_name << endl;
			return;
		}
		
		if (argc == 4)
		{
			// msc[0] = (int)strtol(argv[3], NULL, 10);
			msc[0] = 0;
			msc[1] = (int)strtol(argv[3], NULL, 10);		// when # of input is 4, use it to set end frame
			// reader.set(CAP_PROP_POS_MSEC, msc[0]);  
		}
		else if (argc == 5)
		{
			msc[0] = (int)strtol(argv[3], NULL, 10);
			msc[1] = (int)strtol(argv[4], NULL, 10);
			// reader.set(CAP_PROP_POS_MSEC, msc[0]);  
			reader.set(CV_CAP_PROP_POS_FRAMES, msc[0]);  // set nframe instead of time in ms
			 
		}
		else
		{
			msc[0] = 0;
			msc[1] = 0;
		}
	}
	else
	{
		cout << "file format is not supported." << endl;
		return;
	}	
}

string x2str(int num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(float num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(double num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
string x2str(bool num)
{
	stringstream ss;
	ss << num;
	return ss.str();
}
