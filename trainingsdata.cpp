#include <string>
#include <stdio.h>
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"

using namespace std;
using namespace cv;

const string datasetFolders[4] = { "Dataset/01/", "Dataset/02/", "Dataset/03/", "Dataset/04/" };

void generateLabels(vector<bool>& isRoad, int totalBlocks){
	int isRoadThreshold = (totalBlocks-1)*0.6; // First/upper 60% of blocks is likely not road
	isRoad = vector<bool>(totalBlocks);

	for(int blkIdx=0;blkIdx<totalBlocks;blkIdx++)
		isRoad[blkIdx] = (blkIdx>isRoadThreshold);
}

static void onMouse( int event, int x, int y, int, void* param){
    if( event != EVENT_LBUTTONDOWN )
        return;

    int* coordinates = (int*) param;
    coordinates[0] = x;
    coordinates[1] = y;
}

void guiLabeling(LbpFeatureVector& fv, Mat& img, vector<bool>& isRoad){
	/* Initial configuration */
    ostringstream strBldr; // Stringbuilder
	int blkX,blkY,imWidth,imHeight,outerMargin,innerMargin,blkSize; // Necessary parameters for block coordinates and size
	imWidth = img.cols;
	imHeight = img.rows;
	outerMargin = fv.getOuterMargin();
	innerMargin = fv.getInnerMargin();
	blkX = blkY = outerMargin;
	blkSize = fv.getBlkSize();

	/* Retrieve and draw initial block labeling */
	for(int blkIdx=0; blkIdx<isRoad.size() && fv.validateBlkCoordinates(blkX,blkY,imWidth,imHeight); blkIdx++,fv.incrementBlkCoordinates(blkX,blkY,imWidth)){
		/* label values > 0 are assumed to indicate this block is road */
		isRoad[blkIdx] = (isRoad[blkIdx]>0);

		/* Draw rectangle with block index */
		int red=0,green=0;
		isRoad[blkIdx]? green=255 : red=255;
		rectangle(img
		    ,Point(blkX,blkY)
		    ,Point(blkX+blkSize,blkY+blkSize)
		    ,Scalar(0,green,red)
		);

		Point blkCenter(blkX+blkSize/2-5,blkY+blkSize/2+5);
		strBldr.str(""); strBldr<<blkIdx;
		putText(img,strBldr.str(),blkCenter,FONT_HERSHEY_PLAIN,1,Scalar(0,green,red));
	}

	/* Relabel blocks manually
	* Clicking in a block flips the label
	* [Escape] ends labeling and saves current labels
	*/
	int input = -1; // Keyboard input
	int coordinates[2] = {-1,-1}; // Clicked coordinates to flip label
	int blksInWidth = (imWidth-2*outerMargin)/(innerMargin+blkSize); // Helper value
	string windowName = "Relabeling";
	io::showImage(windowName,img,false);
	setMouseCallback(windowName,onMouse,coordinates);
	while( input != io::KEY_ESCAPE){

		if(coordinates[0] > -1){
			/* Calculate block index from click coordinates */
			int x = coordinates[0];
			int y = coordinates[1];

			int blkIdxWidth = (x-outerMargin)/(blkSize+innerMargin);
			int blkIdxHeight = (y-outerMargin)/(blkSize+innerMargin);

			int blkIdx = blkIdxHeight * blksInWidth + blkIdxWidth;

			int blkX = outerMargin + blkIdxWidth*(blkSize+innerMargin);
			int blkY = outerMargin + blkIdxHeight*(blkSize+innerMargin);

			if( x > blkX && y > blkY && x < blkX+blkSize && y < blkY+blkSize){ // Ignore clicks outside of block
				/* Flip label value */
				isRoad[blkIdx] = !isRoad[blkIdx];

				/* Repaint img */
				int red=0,green=0;
				isRoad[blkIdx]? green=255 : red=255;
				rectangle(img
				    ,Point(blkX,blkY)
				    ,Point(blkX+blkSize,blkY+blkSize)
				    ,Scalar(0,green,red)
				);
			}

			coordinates[0] = -1;
		}

		io::showImage(windowName,img,false);
		input=waitKey(50);
	}
}

void saveFrameOutput(const string fnFrame, const Mat& frame, const Mat& featureVectors, vector<bool>& isRoad){
	vector<short> labels(isRoad.size());
	for(int i=0;i<isRoad.size();i++)
		labels[i] = (isRoad[i]? 1 : -1);

	size_t pos1 = fnFrame.find_last_of("/")+1;
	if(pos1 == string::npos) pos1=0;
	size_t pos2 = fnFrame.find_last_of(".");
	if(pos2 == string::npos) pos2=fnFrame.size();
	string frameName = fnFrame.substr(pos1,pos2-pos1); // Remove folder prefixes and file extension

	io::checkDir(io::dirOutput);

	/* Open new CSV-files for frame data in output directory */
	ofstream fosLbl, fosFv;
	string fnLbl = io::dirOutput+frameName+"_labels.csv";
	string fnFv = io::dirOutput+frameName+"_featurevectors.csv";
	fosLbl.open(fnLbl.c_str(),ios::out);
	fosFv.open(fnFv.c_str(),ios::out);

	for(int blkIdx=0;blkIdx<featureVectors.rows;blkIdx++){
		/* Write label */
		fosLbl<<blkIdx<<","<<labels[blkIdx]<<endl;

		/* Retrieve feature vector from matrix */
		const int* fv_start = featureVectors.ptr<int>(blkIdx);
		const int* fv_end = fv_start + featureVectors.cols;
		vector<int> hist(fv_start,fv_end);

		/* Write feature vector */
		fosFv<<blkIdx<<",\"";
		io::printVector(fosFv,hist);
		fosFv<<"\""<<endl;
	}
	fosLbl.close();
	fosFv.close();

	io::saveImage(frameName+"_blocks",frame);
}

void parameterIteration(){
    int outerMargin(16),lbpRadius(1),histBins(128);
    int innerMargin[] = {64};
    int blkSize[] = {8,16,32};
    Mat frame = imread(datasetFolders[0]+"frame00000.png");

    for(int iMIdx=0;iMIdx<sizeof(innerMargin)/sizeof(int);iMIdx++){
    for(int bSIdx=0;bSIdx<sizeof(blkSize)/sizeof(int);bSIdx++){
	    LbpFeatureVector fv(outerMargin,innerMargin[iMIdx],blkSize[bSIdx],lbpRadius,histBins);
	    Mat featureVectors;

	    char buffer[30];
	    sprintf(buffer,"%02dframe%05d_%03d_%02d.png",1,0,innerMargin[iMIdx],blkSize[bSIdx]);
	    string fnFrame = string(buffer);
	    fv.processFrame(fnFrame,frame,featureVectors,true);

	    vector<bool> isRoad;
	    generateLabels(isRoad,featureVectors.rows);

	    Mat frameWithBlocks;
	    frame.copyTo(frameWithBlocks);
	    guiLabeling(fv,frameWithBlocks,isRoad);

	    saveFrameOutput(fnFrame,frameWithBlocks,featureVectors,isRoad);

	    destroyAllWindows();
	}
	}
}

void fullTraining(bool relabel=false){
    int outerMargin(16),innerMargin(64);
    LbpFeatureVector fv(outerMargin,innerMargin);

	vector<Mat> frames;
	string frameRegex = "frame%05d.png";
	for(int s=0; s<sizeof(datasetFolders)/sizeof(string); s++) {
		io::readImages(datasetFolders[s],frameRegex,frames);
		for(int i=0;i<frames.size();i+=5){
			char buffer[14];
			sprintf(buffer,("%02d"+frameRegex).c_str(),s+1,i);
			string fnFrame = string(buffer);
			Mat featureVectors;
			fv.processFrame(fnFrame,frames[i],featureVectors,true);

			vector<bool> isRoad;
			if(relabel)
				guiLabeling(fv,frames[i],isRoad);
			else
				generateLabels(isRoad,featureVectors.rows);

			saveFrameOutput(fnFrame,frames[i],featureVectors,isRoad);
		}
	}
}

int main (int argc, char** argv){
	parameterIteration();
	// fullTraining();

    return 0;
}