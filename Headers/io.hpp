#ifndef IO_
#define IO_

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

namespace io {

	/** Directories **/
	string dirOutput = "output/";
	string dirTrainingsdata = "Trainingsdata/";

	/** Shown image properties and helper values **/
	int imX=50,imY=50
		,shownImages=0
		,consecutiveImages=6
		,normalSize=325;

	void showImage(string windowName,Mat& img, bool resize=true, int imWidth=0, int imHeight=0){
	    int windowFlag, xSize, ySize, margin=3;
	    if(resize){
	        windowFlag = WINDOW_NORMAL;
	        xSize = ySize = normalSize;
	    } else {
	        windowFlag = WINDOW_AUTOSIZE;
	        xSize = imWidth;
	        ySize = imHeight;
	    }

	    namedWindow(windowName,windowFlag);
	    imshow(windowName, img);
	    moveWindow(windowName,imX,imY);
	    imX += xSize+margin;
	    if(++shownImages%consecutiveImages == 0){
	    	imY += ySize+margin;
	    }
	}

	void saveImage(string fn, Mat& img){
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

	void readTrainingsdata(vector<short>& labels, vector<vector<int> >& featureVectors){
		for(int s=1;s<=1;s++) {
		    char folder[36];
		    sprintf(folder, "%02d",s);
			for(int f=0; f<=200; f+=5) {
			    char number[36];
			    sprintf(number, "%05d", f);
			    string fnLbl = dirTrainingsdata+string(folder)+"frame"+string(number)+"_labels.csv";
			    string fnFv = dirTrainingsdata+string(folder)+"frame"+string(number)+"_featurevectors.csv";
			    if(file_exists(fnLbl) && file_exists(fnFv)) {
					cout<<"Reading trainingsdata of frame "<<string(folder)+"frame"+string(number)<<endl;
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
	}
}

#endif