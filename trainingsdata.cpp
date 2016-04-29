#include <string>
#include <stdio.h>
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"

using namespace std;
using namespace cv;

int main (int argc, char** argv){
	string datasetFolder = "Dataset/01/";

	if(argc == 2)
		datasetFolder = argv[1];

    /* Training data */
    int outerMargin(16),innerMargin(64);
    LbpFeatureVector fv(outerMargin,innerMargin);

	vector<Mat> frames;
	string frameRegex = "frame%05d.png";
	io::readImages(datasetFolder,frameRegex,frames);
	for(int i=0;i<frames.size();i+=5){
		char fnFrame[14];
		sprintf(fnFrame,frameRegex.c_str(),i);
		Mat featureVectors;
		fv.processFrame(string(fnFrame),frames[i],featureVectors,true);
	}

    destroyAllWindows();
    return 0;
}