#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	VideoCapture capframes("./Dataset/01/frame%05d.png");
	vector<Mat> frames;
	while( capframes.isOpened() )
	{
	    Mat img;
	    if(!capframes.read(img)) {
	    	break;
	    }
	   	frames.push_back(img);
	}

	VideoCapture capmasks("./Dataset/01/mask%05d.png");	
	vector<Mat> masks;
	while( capmasks.isOpened() )
	{
	    Mat img;
	    if(!capmasks.read(img)) {
	    	break;
	    }
	    capmasks >> img;
	   	masks.push_back(img);
	}

	double alpha = 0.7;
	double beta = 1.0 - alpha;
	namedWindow( "Display window", WINDOW_AUTOSIZE );
	for(int i = 0; i<frames.size(); i++) {
		Mat dst;
		cvtColor(frames[i], frames[i], CV_RGB2GRAY);
		addWeighted( masks[i] , alpha, frames[i], beta, 0.0, dst);																																																																																																																																																																																																																				
		imshow( "Display window", dst );  
		waitKey(0);
	}

	waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							