#include <string>
#include <stdio.h>
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"

using namespace std;
using namespace cv;

int main (int argc, char** argv){
	string datasetFolders[4] = { "Dataset/01/", "Dataset/02/", "Dataset/03/", "Dataset/04/" };

    /* Training data */
    int outerMargin(16),innerMargin(64);
    LbpFeatureVector fv(outerMargin,innerMargin);

	vector<Mat> frames;
	string frameRegex = "frame%05d.png";
	for(int s=0; s<sizeof(datasetFolders)/sizeof(string); s++) {
		io::readImages(datasetFolders[s],frameRegex,frames);
		for(int i=0;i<frames.size();i+=5){
			char fnFrame[14];
			sprintf(fnFrame,frameRegex.c_str(),i);
			string regex = "%02d"+string(fnFrame);
			sprintf(fnFrame,regex.c_str(),s+1);
			Mat featureVectors;
			fv.processFrame(string(fnFrame),frames[i],featureVectors,true);
		}
	    destroyAllWindows();
	}
    return 0;
}