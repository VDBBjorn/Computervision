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
	   	matrices.push_back(img);
	}
	return matrices;
}

void showMaxSpeed(vector<Mat> & masks) {
    namedWindow( "Display window", WINDOW_AUTOSIZE );
	for(int i = 0; i<masks.size(); i++) {
        Point pt;
        pt.x = 600;
        pt.y = 405;
        Scalar color = Scalar( 255,0,0 );
        circle( masks[i], pt, 2, color);
        imshow( "Display window", masks[i] );
        cv::Vec3b vec = masks[i].at<cv::Vec3b>(pt.y,pt.x);
        float max = vec[0]/255.0 *100.0 * 3.6;
        cout << vec << " => " << max << "km/u" << endl;
		waitKey(0);
	}
}

int main(int argc, char** argv){
	if(argc <= 1) {
		cout << "add parameters: dataset folder" << endl;
		return 1;
	}
    
	vector<Mat> masks = read_images(argv[1],"mask%05d.png");

	showMaxSpeed(masks);

	waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							