#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "Headers/io.hpp"
#include "Headers/LineDetection.hpp"
#include <iomanip>


using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if(argc <= 1) {
		cout << "add parameters: dataset folder" << endl;
		return 1;
	}	

	vector<Mat> frames,masks;
	io::readImages(argv[1],"frame%05d.png",frames);
	io::readImages(argv[1],"mask%05d.png",masks);

	// detect lines
	LineDetection ld;
	for(int i=0; i < frames.size(); i++) {
		Mat lines = ld.getLinesFromImage(frames[i], false);
		//waitKey(0);
		io::checkDir("linetest");
		stringstream ssOut;
		ssOut << "linetest/frame" << setfill('0') << std::setw(5) << i << ".png";
		imwrite(ssOut.str(),lines);

		destroyAllWindows();
	}

	waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							

