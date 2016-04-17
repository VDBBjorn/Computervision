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
}

#endif