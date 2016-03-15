#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;

vector<Mat> read_images(String folder, String regex) {
	VideoCapture cap(folder+"/"+regex);
	vector<Mat> matrices;
	while( cap.isOpened() )
	{
	    Mat img;
	    if(!cap.read(img)) {
	    	return matrices;
	    }
		if(img.channels()>1) cvtColor(img, img, CV_RGB2GRAY);
	   	matrices.push_back(img);
	}
	return matrices;
}

void showBlendedImages(vector<Mat> & frames, vector<Mat> & masks) {
	double alpha = 1.0;
	double beta = 1.0;
	namedWindow( "Display window", WINDOW_AUTOSIZE );
	for(int i = 0; i<frames.size(); i++) {
		Mat dst;
		addWeighted( masks[i] , alpha, frames[i], beta, 0.0, dst);
		imshow( "Display window", dst );  
		waitKey(0);
	}
}

int main(int argc, char** argv){
	if(argc <= 1) {
		cout << "add parameters: dataset folder" << endl;
		return 1;
	}	

	vector<Mat> frames = read_images(argv[1],"frame%05d.png");
	vector<Mat> masks = read_images(argv[1],"mask%05d.png");

	showBlendedImages(frames, masks);

	waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							