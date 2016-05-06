#ifndef IO_
#define IO_

#include <string>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

namespace io {

	/** Constants **/
	const string dirOutput = "output/";
	const string dirTrainingsdata = "Trainingsdata/";
	const string labelsPostfix = "_labels.csv";
	const string featVecsPostfix = "_featurevectors.csv";
	const int KEY_ESCAPE = 537919515;
	const int KEY_ENTER = 537919498;
	const int KEY_B = 537919498;

	/** Shown image properties and helper values **/
	int imX=50,imY=50
		,shownImages=0
		,consecutiveImages=6
		,normalSize=325;

	map<string,vector<int> > imgPos;

	void showImage(string windowName,Mat& img, bool resize=false){
	    int windowFlag, xSize, ySize, margin=3;
	    if(resize){
	        windowFlag = WINDOW_NORMAL;
	        xSize = ySize = normalSize;
	    } else {
	        windowFlag = WINDOW_AUTOSIZE;
	        xSize = img.cols;
	        ySize = img.rows;
	    }

	    namedWindow(windowName,windowFlag);
	    imshow(windowName, img);
	    if(imgPos[windowName].empty()){
	    	imgPos[windowName] = vector<int>(2);
	    	imgPos[windowName][0]=imX;
	    	imgPos[windowName][1]=imY;
	    	moveWindow(windowName,imX,imY);
	    }
	    imX += xSize+margin;
	    if(++shownImages%consecutiveImages == 0){
	    	imX = 50;
	    	imY += ySize+margin;
	    }
	}

	void saveImage(const string fn, const Mat& img){
	    imwrite(dirOutput+fn+".png",img);
	}

	template <typename T>
	void printVector(ostream& os, const vector<T>& v){    int i=0;
	    while(i<v.size()-1){
	        os<<v[i]<<";";
	        i++;
	    }
	    os<<v[i];
	}

	void checkDir(string dir){
	    struct stat st;
	    const char* cDir = dir.c_str();
	    if(stat(cDir,&st) == -1){
	        if(mkdir(cDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH ) != 0){
	            cerr<<"Error in checkOutputDir : failed to create dir "<<dir<<endl;
	            throw;
	        }
	    }
	}

	void readImages(String folder, String regex, vector<Mat>& images) {
		VideoCapture cap(folder+"/"+regex);
		images.clear();
		while( cap.isOpened() )
		{
		    Mat img;
		    if(!cap.read(img)) {
		    	return;
		    }
		   	images.push_back(img);
		}
	}

	void showBlendedImages(vector<Mat> & frames, vector<Mat> & masks) {
		double alpha = 1.0;
		double beta = 1.0;
		namedWindow( "Display window", WINDOW_AUTOSIZE );
		for(int i = 0; i<frames.size(); i++) {
			Mat dst;
			if(frames[i].channels()>1) cvtColor(frames[i], frames[i], CV_RGB2GRAY);
			if(masks[i].channels()>1) cvtColor(masks[i], masks[i], CV_RGB2GRAY);
			addWeighted( masks[i] , alpha, frames[i], beta, 0.0, dst);
			imshow( "Display window", dst );  
			waitKey(0);
		}
	}

	bool file_exists(const string& name) {
		struct stat buffer;   
		return (stat (name.c_str(), &buffer) == 0); 
	}

	void readTrainingsdata(vector<short>& datasets, Mat& labelsMat, Mat& trainingsMat){
		vector<short> labels;
		vector<vector<int> > featureVectors;

		for(int s=0;s<datasets.size();s++) {
			for(int f=0; f<=200; f+=5) {
			    char buffer[36];
			    sprintf(buffer, "%02dframe%05d",datasets[s],f);
			    string frameName = string(buffer);
			    string fnLbl = dirTrainingsdata+frameName+labelsPostfix;
			    string fnFv = dirTrainingsdata+frameName+featVecsPostfix;
			    if(file_exists(fnLbl) && file_exists(fnFv)) {
					cout<<"Reading trainingsdata of frame "<<frameName<<endl;
				 	ifstream ifsLbl(fnLbl.c_str());
				 	ifstream ifsFv(fnFv.c_str());

					string lineLbl,lineFv;
					while( getline(ifsLbl,lineLbl) && getline(ifsFv,lineFv) ){
						string strLbl = lineLbl.substr(lineLbl.find_first_of(',')+1);
						labels.push_back(atoi(strLbl.c_str()));

						size_t pos = lineFv.find_first_of('"')+1;
						string strFv = lineFv.substr(pos);
						stringstream ssFv(strFv);
						vector<int> fvValues;
						string strValue;
						while(getline(ssFv,strValue,';')) {
							fvValues.push_back(atoi(strValue.c_str()));
						}
						featureVectors.push_back(fvValues);
					}
				}
			}
		}

		labelsMat = Mat::zeros(labels.size(), 1, CV_32SC1);
	 	for(int i=0; i<labels.size();i++) {
			labelsMat.at<int>(i,0) = labels[i];
		}
		trainingsMat = Mat::zeros(featureVectors.size(), featureVectors[0].size(), CV_32FC1);
		for(int i = 0; i<featureVectors.size(); i++) {
			for(int j=0; j<featureVectors[i].size(); j++) {
				trainingsMat.at<float>(i,j) = featureVectors[i][j];
			}
		}
	}

	void readTrainingsdataOutput(string frameName, Mat& labelsMat, Mat& trainingsMat){
		vector<short> labels;
		vector<vector<int> > featureVectors;

	    string fnLbl = dirOutput+frameName+labelsPostfix;
	    string fnFv = dirOutput+frameName+featVecsPostfix;
	    if(file_exists(fnLbl) && file_exists(fnFv)) {
			cout<<"Reading trainingsdata from output of frame "<<frameName<<endl;
		 	ifstream ifsLbl(fnLbl.c_str());
		 	ifstream ifsFv(fnFv.c_str());

			string lineLbl,lineFv;
			while( getline(ifsLbl,lineLbl) && getline(ifsFv,lineFv) ){
				string strLbl = lineLbl.substr(lineLbl.find_first_of(',')+1);
				labels.push_back(atoi(strLbl.c_str()));

				size_t pos = lineFv.find_first_of('"')+1;
				string strFv = lineFv.substr(pos);
				stringstream ssFv(strFv);
				vector<int> fvValues;
				string strValue;
				while(getline(ssFv,strValue,';')) {
					fvValues.push_back(atoi(strValue.c_str()));
				}
				featureVectors.push_back(fvValues);
			}
		}

		size_t oldRowSize = labelsMat.rows;
		if(oldRowSize==0){
			labelsMat = Mat::zeros(labels.size(), 1, CV_32SC1);
			trainingsMat = Mat::zeros(featureVectors.size(), featureVectors[0].size(), CV_32FC1);
		}else{
			labelsMat.resize(oldRowSize+labels.size(),0);
			trainingsMat.resize(oldRowSize+labels.size(),0);
		}
	 	for(int i=0; i<labels.size();i++) {
			labelsMat.at<int>(i,0) = labels[i];
			for(int j=0; j<featureVectors[i].size(); j++) {
				trainingsMat.at<float>(i,j) = featureVectors[i][j];
			}
		}
	}

	void readFrameLabels(string fnLbl, vector<bool>& isRoad){
		if(!file_exists(fnLbl)){
			cerr<<"Trying to read file that doesn't exist: "<<fnLbl<<endl;
			throw;
		}

	 	ifstream ifsLbl(fnLbl.c_str());
	 	string lineLbl;
	 	while( getline(ifsLbl,lineLbl) ){
			string strLbl = lineLbl.substr(lineLbl.find_first_of(',')+1);
			isRoad.push_back(
				( atoi(strLbl.c_str()) > 0 )
			);
	 	}
	}
}

#endif