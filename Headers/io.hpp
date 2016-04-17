#ifndef IO_
#define IO_

#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

namespace io {

	/** Output directory **/
	string dirOutput = "output/";

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

	void read_images(String folder, String regex, vector<Mat>& images) {
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
}

#endif