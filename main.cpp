#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include "LineDetection/LineDetection.cpp"
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if(argc <= 1) {
		cout << "add parameters: dataset folder" << endl;
		return 1;
	}	

	vector<Mat> frames,masks;
	io::read_images(argv[1],"frame%05d.png",frames);
	io::read_images(argv[1],"mask%05d.png",masks);

	// detect lines
	// TODO: Filter lines in wrong orientation and choose better parameters
	LineDetection ld;
	int initialHoughVote = 100;
	int initialHoughVote2 = 100;
	int houghVote = initialHoughVote;
	int houghVote2 = initialHoughVote2;
	for(int i=0; i < frames.size(); i++) {
		bool drawLines = true; // draw detected lines on source image
		bool debugLinedetection = false; // wait after each frame and show all intermediate results
		vector<RoadLine> lines = ld.getLinesFromImage(frames[i], initialHoughVote, houghVote,initialHoughVote2, houghVote2, drawLines, debugLinedetection);
	}

	io::showBlendedImages(frames, masks);

	waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							

