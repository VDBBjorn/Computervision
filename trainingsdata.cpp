#include <string>
#include <stdio.h>
// #include <time.h>
#include <set>
#include "Headers/io.hpp"
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/svm.hpp"

using namespace std;
using namespace cv;

void generateLabels(string frameName, vector<bool>& isRoad, int totalBlocks){
	string fnLabels(io::dirOutput+frameName+io::labelsPostfix);
	if(io::file_exists(fnLabels)){
		io::readFrameLabels(fnLabels,isRoad);
	}else{
		int isRoadThreshold = (totalBlocks-1)*0.6; // First/upper 60% of blocks is likely not road
		isRoad = vector<bool>(totalBlocks);

		for(int blkIdx=0;blkIdx<totalBlocks;blkIdx++)
			isRoad[blkIdx] = (blkIdx>isRoadThreshold);
	}
}

static void onMouse( int event, int x, int y, int, void* param){
    if( event != EVENT_LBUTTONDOWN )
        return;

    int* coordinates = (int*) param;
    coordinates[0] = x;
    coordinates[1] = y;
}

void fileToFrameName(string fnFrame, string& frameName){
	size_t pos1 = fnFrame.find_last_of("/")+1;
	if(pos1 == string::npos) pos1=0;
	size_t pos2 = fnFrame.find_last_of(".");
	if(pos2 == string::npos) pos2=fnFrame.size();
	frameName = fnFrame.substr(pos1,pos2-pos1); // Remove folder prefixes and file extension
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

	/* Draw initial block labeling */
	for(int blkIdx=0; 
			blkIdx<isRoad.size() && fv.validateBlkCoordinates(blkX,blkY,imWidth,imHeight);
			blkIdx++,fv.incrementBlkCoordinates(blkX,blkY,imWidth)){
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
	int blksInWidth = (imWidth-2*outerMargin-blkSize)/(innerMargin+blkSize) + 1; // Helper value
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

				/* Repaint block and index on img */
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

			coordinates[0] = -1;
		}

		io::showImage(windowName,img,false);
		input=waitKey(50);
	}
}

void saveFrameOutput(const string frameName, const Mat& frame, const Mat& featureVectors, vector<bool>& isRoad, bool useLBP, bool useColor){
	vector<short> labels(isRoad.size());
	for(int i=0;i<isRoad.size();i++)
		labels[i] = (isRoad[i]? 1 : -1);

	io::checkDir(io::dirOutput);

	/* Open new CSV-files for frame data in output directory */
	ofstream fosLbl, fosFv;
	string fnLbl = io::dirOutput+frameName+io::labelsPostfix;
	string fnFv = io::dirOutput+frameName+(useLBP?io::lbpStr:"")+(useColor?io::colorStr:"")+io::featVecsPostfix;
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

	if(!frame.empty()) io::saveImage(frameName+"_blocks",frame);
}

void output_to_csv_header(ofstream & csv, bool useColor, bool useLBP, bool includeMarks) {
	csv<< endl;
	csv<<"FV w Color;FV w LBP;include road marks"<<endl;
	csv<<(useColor?"Yes":"No")<<";"<<(useLBP?"Yes":"No")<<";"<<(includeMarks?"Yes":"No")<<";"<<endl;
	csv<<"C;gamma;blkSize;datasets trained on;TP;FN;FP;TN;precision;accuracy;recall;True negative rate;F"<<endl;
}

void output_to_csv(ofstream & csv, set<int>& trainingsSet, my_svm & svm, int blkSize, Mat & testLabels, Mat & testTrainingsdata) {
	double c = (svm.get_svm())->getC();
	// Mat classWeights = svm.get_svm()->getClassWeights();
	// string classWeightsString;
	// io::convertMatToString(classWeights,classWeightsString);
	// double coef0 = svm.get_svm()->getCoef0();
	// double degree = svm.get_svm()->getDegree();
	double gamma = svm.get_svm()->getGamma();
	// double nu = svm.get_svm()->getNu();
	// double p = svm.get_svm()->getP();
	svm.calculateScores(testLabels,testTrainingsdata);
	Mat* confusion = svm.get_confusion_matrix();
	int TP = confusion->at<int>(0,0);
	int FN = confusion->at<int>(0,1);
	int FP = confusion->at<int>(1,0);
	int TN = confusion->at<int>(1,1);
	ostringstream strBldr;
	strBldr<<"\"";
	set<int>::iterator it;
	for(it = trainingsSet.begin(); it!=trainingsSet.end(); it++){
		strBldr<<(*it)<<",";
	}
	string trainingsSetString = strBldr.str();
	trainingsSetString = trainingsSetString.substr(0,trainingsSetString.size()-1);
	trainingsSetString+="\"";
	
	csv<<c<<";"<<gamma<<";"<<blkSize<<";"<<trainingsSetString<<";"<<TP<<";"<<FN<<";"<<FP<<";"<<TN<<";"<<svm.get_precision()<<";"<<svm.get_accuracy()<<";"<<svm.get_recall()<<";"<<svm.get_true_negative()<<";"<<svm.get_F()<<endl;
}

void parameterIterationTraining(bool relabel=true){
	relabel = false;

    int outerMargin(16),lbpRadius(1),histBins(128);
    int innerMargins[] = {64};
    int blkSizes[] = {8,16,32}; //{8,16,32};
    int datasets[] = {1,2,3,4}; //{1,2,3,4};
    bool datasetDoubles = false;
    int frameInterval = 10;
    // int frameStopIdx = io::FRAME_MAX_IDX;
    int frameStopIdx = 50;
	bool trainAuto = false; // Whether or not to use automatic training for SVM
	bool includeMarksVals[] = {false,true};
	bool useColorVals[] = {false,true,true};
	bool useLBPVals[] = {true,false,true};

	string frameName;
	char buffer[30];

    /* Iterate different parameters for each frame, process the frame and save output to file. */
	for(int dSIdx=0;dSIdx<sizeof(datasets)/sizeof(int);dSIdx++){
	int dataset = datasets[dSIdx];
	for(int fIdx=0;fIdx<frameStopIdx;fIdx+=frameInterval){

		string fnFrame = io::datasetFolders[dataset-1]+"frame%05d.png";
		sprintf(buffer,fnFrame.c_str(),fIdx);
		fnFrame = string(buffer);

		if(!io::file_exists(fnFrame)) continue;
	    Mat frame = imread(fnFrame);

	    for(int iMIdx=0;iMIdx<sizeof(innerMargins)/sizeof(int);iMIdx++){
		int innerMargin = innerMargins[iMIdx];
	    for(int bSIdx=0;bSIdx<sizeof(blkSizes)/sizeof(int);bSIdx++){
		int blkSize = blkSizes[bSIdx];
		for(int incMIdx=0;incMIdx<sizeof(includeMarksVals)/sizeof(bool);incMIdx++){
		bool includeMarks = includeMarksVals[incMIdx];
		for(int fvTypeIdx=0;fvTypeIdx<sizeof(useLBPVals)/sizeof(bool);fvTypeIdx++){
		bool useLBP = useLBPVals[fvTypeIdx], useColor = useColorVals[fvTypeIdx];

		    LbpFeatureVector fv(outerMargin,innerMargin,blkSize,lbpRadius,histBins);
		    Mat featureVectors;

		    io::buildFrameName(buffer,frameName,dataset,fIdx,innerMargin,blkSize,includeMarks);
		    fv.processFrame(frameName,frame,featureVectors,useLBP,useColor,true);

		    vector<bool> isRoad;
		    generateLabels(frameName,isRoad,featureVectors.rows);

		    Mat frameWithBlocks;
		    if(relabel){
			    frame.copyTo(frameWithBlocks);
			    guiLabeling(fv,frameWithBlocks,isRoad);
			}

		    saveFrameOutput(frameName,frameWithBlocks,featureVectors,isRoad,useLBP,useColor);

		    destroyAllWindows();
		}
		}
		}
		}
	}
	}

	/* Iterate framesets per combination of parameters.
	* Train SVM with one part of the frames, test SVM with the other part.
	* Calculate confusion matrix and F-scores, save results to file.
	*/
	vector<set<int> > trainingsSets;
	int nDatasets = sizeof(datasets)/sizeof(int);
	bool isTrainingSet;

	io::checkDir(io::dirOutputLogging);
	ofstream csv;
	string fnFrameOutput = io::dirOutputLogging+"output_"+io::currentDateTime()+".csv";
	csv.open(fnFrameOutput.c_str(),ios::out);

	bool useCoupleTrainingsSets = true;
	bool useTripletTrainingsSets = true;
	if(useCoupleTrainingsSets){
		for(int i=0; i<nDatasets-1; i++){
			for(int j=i+1;j<nDatasets;j++){
				set<int> tS;
				tS.insert(datasets[i]);
				tS.insert(datasets[j]);
				trainingsSets.push_back(tS);
			}
		}
	}
	if(useTripletTrainingsSets){
		for(int i=0; i<nDatasets-2; i++){
			for(int j=i+1;j<nDatasets-1;j++){
				for(int k=j+1;k<nDatasets;k++){
					set<int> tS;
					tS.insert(datasets[i]);
					tS.insert(datasets[j]);
					tS.insert(datasets[k]);
					trainingsSets.push_back(tS);
				}
			}
		}
	}

	if(trainingsSets.empty() && nDatasets > 0){ // Single dataset
		set<int> tS;
		tS.insert(datasets[0]);
		tS.insert(datasets[0]);
		trainingsSets.push_back(tS);
	}

	for(int incMIdx=0;incMIdx<sizeof(includeMarksVals)/sizeof(bool);incMIdx++){
	bool includeMarks = includeMarksVals[incMIdx];
	for(int fvTypeIdx=0;fvTypeIdx<sizeof(useLBPVals)/sizeof(bool);fvTypeIdx++){
	bool useLBP = useLBPVals[fvTypeIdx], useColor = useColorVals[fvTypeIdx];

		output_to_csv_header(csv,useColor,useLBP,includeMarks);

	    for(int iMIdx=0;iMIdx<sizeof(innerMargins)/sizeof(int);iMIdx++){
	    int innerMargin = innerMargins[iMIdx];
	    for(int bSIdx=0;bSIdx<sizeof(blkSizes)/sizeof(int);bSIdx++){
	    int blkSize = blkSizes[bSIdx];
	    for(int tSIdx=0;tSIdx<trainingsSets.size();tSIdx++){
	    set<int>& trainingsSet = trainingsSets[tSIdx];

	    	// Read trainings and test data for trainings set
	    	Mat initLabels, initTrainingsdata, testLabels, testTrainingsdata;
			for(int dSIdx=0;dSIdx<sizeof(datasets)/sizeof(int);dSIdx++){
			int dataset = datasets[dSIdx];
			for(int fIdx=0;fIdx<frameStopIdx;fIdx+=frameInterval){
				io::buildFrameName(buffer,frameName,dataset,fIdx,innerMargin,blkSize,includeMarks);
				isTrainingSet = trainingsSet.find(dataset)!=trainingsSet.end();
				if(isTrainingSet)
				    io::readTrainingsdataOutput(frameName,initLabels,initTrainingsdata,useLBP,useColor);
				if(!isTrainingSet || datasetDoubles)
				    io::readTrainingsdataOutput(frameName,testLabels,testTrainingsdata,useLBP,useColor);
			}
			}
	    	// Init SVM and train
		    my_svm svm(initLabels,initTrainingsdata,trainAuto);

		    // Use SVM on different test data than trained with, save confusion matrix and F-scores.
		    output_to_csv(csv, trainingsSet, svm, blkSize, testLabels, testTrainingsdata);
	    }
	    }
		}
	}
	}
	csv.close();
}

void fullTraining(bool relabel=false){
 //    LbpFeatureVector fv(io::outerMargin,io::innerMargin);

	// vector<Mat> frames;
	// string frameRegex = "frame%05d.png";
	// for(int s=0; s<sizeof(io::datasetFolders)/sizeof(string); s++) {
	// 	io::readImages(io::datasetFolders[s],frameRegex,frames);
	// 	for(int i=0;i<frames.size();i+=5){
	// 		char buffer[30];
	// 		sprintf(buffer,("%02d"+frameRegex).c_str(),s+1,i);
	// 		string fnFrame = string(buffer);
	// 		Mat featureVectors;
	// 		fv.processFrame(fnFrame,frames[i],featureVectors,true,true,true);

	// 		string frameName;
	// 		fileToFrameName(fnFrame,frameName);

	// 		vector<bool> isRoad;
	// 		generateLabels(frameName,isRoad,featureVectors.rows);
	// 		if(relabel)	guiLabeling(fv,frames[i],isRoad);

	// 		saveFrameOutput(frameName,frames[i],featureVectors,isRoad);
	// 	}
	// }
}

int main (int argc, char** argv){
	parameterIterationTraining();
	// fullTraining();

    return 0;
}