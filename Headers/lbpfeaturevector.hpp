#ifndef FEATURE_VECTORS_
#define FEATURE_VECTORS_

#include <cstdio>
#include <string>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "io.hpp"

using namespace std;
using namespace cv;

class LbpFeatureVector{
private:
	/** Block configuration **/
	int outerMargin;
	int innerMargin;
	int blkX;
	int blkY;
	int blkSize;

	/** Featurevector configuration **/
	int lbpRadius;
	int histBins;
public:
	//LbpFeatureVector(): outerMargin(1),innerMargin(0),blkSize(32),lbpRadius(1),histBins(128){}

	LbpFeatureVector(int _outerMargin=1,int _innerMargin=0,int _blkSize=32,int _lbpRadius=1,int _histBins=128):outerMargin(_outerMargin),innerMargin(_innerMargin),blkSize(_blkSize),lbpRadius(_lbpRadius),histBins(_histBins){}

	/* Calculate LBP values of pixels in src to dst, with radius r.
	* Only calculate withing given window, measured by source coordinates (x,y) and winSize.
	* Based on lbp code from https://github.com/bytefish/opencv/tree/master/lbp
	*/
	template <typename _Tp>
	void _LBP(const Mat& src, Mat& dst, int r, int x, int y, int winSize) {
	    // if( x<r || y<r || x+winSize+r>src.cols || y+winSize+r>src.rows){
	    //     cerr << "Error in _LBP : source coordinates and window size make radius exceed image size" << endl;
	    //     throw;
	    // }
	    dst = Mat::zeros(winSize, winSize, CV_8UC(src.channels()));
	    vector<Mat> srcChnls(src.channels()) , dstChnls(src.channels());
	    split(src,srcChnls);
	    split(dst,dstChnls);
	    for(int i=y;i<y+winSize;i++) {
	        for(int j=x;j<x+winSize;j++) {
		        for(int k=0;k<srcChnls.size();k++) {
		            _Tp center = srcChnls[k].at<_Tp>(i,j);
		            unsigned char code = 0;
		            code |= (srcChnls[k].at<_Tp>(i-r,j-r) > center) << 7;
		            code |= (srcChnls[k].at<_Tp>(i-r,j) > center) << 6;
		            code |= (srcChnls[k].at<_Tp>(i-r,j+r) > center) << 5;
		            code |= (srcChnls[k].at<_Tp>(i,j+r) > center) << 4;
		            code |= (srcChnls[k].at<_Tp>(i+r,j+r) > center) << 3;
		            code |= (srcChnls[k].at<_Tp>(i+r,j) > center) << 2;
		            code |= (srcChnls[k].at<_Tp>(i+r,j-r) > center) << 1;
		            code |= (srcChnls[k].at<_Tp>(i,j-r) > center) << 0;
		            dstChnls[k].at<unsigned char>(i-y,j-x) = code;
		        }
		    }
	    }
	    merge(&dstChnls[0],dstChnls.size(),dst); // &dstChnls[0] makes the vector look like a pointer to an array
	}

	/*
	* Creates histogram for src
	*/
	template <typename _Tp>
	void _histogram(const Mat& src, vector<int>& hist, int numBins) {
	    hist = vector<int>(numBins);
	    int binWidth = 256/numBins;
	    for(int i = 0; i < src.rows; i++) {
	        for(int j = 0; j < src.cols; j++) {
	            int bin = src.at<_Tp>(i,j)/binWidth;
	            hist[bin] += 1;
	        }
	    }
	}

	/* Wrapper function for LBP_ */
	/* Based on code from https://github.com/bytefish/opencv/tree/master/lbp */
	void LBP(const Mat& src, Mat& dst, int radius, int x, int y, int winSize) {
		const int CHAR 		= CV_8SC(src.channels());
		const int UCHAR 	= CV_8UC(src.channels());
		const int SHORT 	= CV_16SC(src.channels());
		const int USHORT 	= CV_16UC(src.channels());
		const int INT 		= CV_32SC(src.channels());
		const int FLOAT 	= CV_32FC(src.channels());
		const int DOUBLE 	= CV_64FC(src.channels());
		if(src.type()==CHAR){
			_LBP<char>(src, dst, radius, x, y, winSize);
		}else if(src.type()==UCHAR){
			_LBP<unsigned char>(src, dst, radius, x, y, winSize);
		}else if(src.type()==SHORT){
			_LBP<short>(src, dst, radius, x, y, winSize);
		}else if(src.type()==USHORT){
			_LBP<unsigned short>(src, dst, radius, x, y, winSize);
		}else if(src.type()==INT){
			_LBP<int>(src, dst, radius, x, y, winSize);
		}else if(src.type()==FLOAT){
			_LBP<float>(src, dst, radius, x, y, winSize);
		}else if(src.type()==DOUBLE){
			_LBP<double>(src, dst, radius, x, y, winSize);
		}else{
			cerr<<"Error in LBP : convert src to grayscale with valid type"<<endl;throw;
		}
	}

	/* Wrapper function for histogram_ */
	/* Based on code from https://github.com/bytefish/opencv/tree/master/lbp */
	void histogram(const Mat& src, vector<int>& hist, int numBins) {
		const int CHAR 		= CV_8SC(src.channels());
		const int UCHAR 	= CV_8UC(src.channels());
		const int SHORT 	= CV_16SC(src.channels());
		const int USHORT 	= CV_16UC(src.channels());
		const int INT 		= CV_32SC(src.channels());
		const int FLOAT 	= CV_32FC(src.channels());
		const int DOUBLE 	= CV_64FC(src.channels());
		if(src.type()==CHAR){
			_histogram<char>(src, hist, numBins);
		}else if(src.type()==UCHAR){
			_histogram<unsigned char>(src, hist, numBins);
		}else if(src.type()==SHORT){
			_histogram<short>(src, hist, numBins);
		}else if(src.type()==USHORT){
			_histogram<unsigned short>(src, hist, numBins);
		}else if(src.type()==INT){
			_histogram<int>(src, hist, numBins);
		}else if(src.type()==FLOAT){
			_histogram<float>(src, hist, numBins);
		}else if(src.type()==DOUBLE){
			_histogram<double>(src, hist, numBins);
		}else{
			cerr<<"Error in histogram : convert src to grayscale with valid type"<<endl;throw;
		}
	}

	/*
	* Generates a featurevector for the image.
	* Histograms of pixel values are calculated per channel and are appended to one another,
	* which makes up the featurevector.
	*/
	void featureVector(const Mat& src,vector<int>& fv){
		vector<Mat> srcChnls(src.channels());
		split(src,srcChnls);
		for(int i=0;i<srcChnls.size();i++){
			vector<int> hist;
			histogram(srcChnls[i],hist,histBins);
			fv.insert(fv.end(),hist.begin(),hist.end());
		}
	}

	/*
	* Processes frame fnFrame to generate featurevectors according to set parameters in this instance of LbpFeatureVector.
	* Each row of featVectors will contain a featurevector of a block.
	* If trainingsdata is ment to be generated, CSV-files containing labels and featurevectors will be created,
	* along with a copy of the frame with the blocks drawn on top of it.
	*/
	void processFrame(string fnFrame, Mat& img, Mat& featVectors, bool isTrainingsdata=false){
    	ostringstream strBldr; // Stringbuilder

    	cout<<"Start processFrame "<<fnFrame<<endl;
    	/* Initial configuration */
		blkX = outerMargin;
		blkY = outerMargin;
		if(outerMargin<lbpRadius) outerMargin = lbpRadius;

		/*    Process frame   */
    	Mat dst;
    	if(isTrainingsdata) img.copyTo(dst);

		size_t pos = fnFrame.find_last_of("/")+1;
		if(pos == string::npos) pos=0;
		string frameName = fnFrame.substr(pos,fnFrame.size()-pos-4); // Remove folder prefixes and file extension
		
		int imWidth=img.cols;
		int imHeight=img.rows;

		io::checkDir(io::dirOutput);

		/* Open new CSV-files for frame data in output directory */
		ofstream fosLbl, fosFv;
		if(isTrainingsdata){
			string fnLbl = io::dirOutput+frameName+"_labels.csv";
			string fnFv = io::dirOutput+frameName+"_featurevectors.csv";
			fosLbl.open(fnLbl.c_str(),ios::out);
			fosFv.open(fnFv.c_str(),ios::out);
		}

		// Number of blocks in width and height
		int numBlksWidth = (imWidth-2*outerMargin-blkSize)/(innerMargin+blkSize) + 1;
		int numBlksHeight = (imHeight-2*outerMargin-blkSize)/(innerMargin+blkSize) + 1;
		int totalBlocks = numBlksWidth*numBlksHeight - 1; // -1 Accounts for 0-indexing
		int isRoadThreshold = totalBlocks*0.6;


		featVectors = Mat(totalBlocks+1,histBins*img.channels(),CV_32SC1);	

		/* Process blocks in frame */
		int blkIdx = -1;
		while(blkX+blkSize+outerMargin < imWidth && blkY+blkSize+outerMargin < imHeight){
		    blkIdx++;

		    /* Calculate LBP values for current block */
		    Mat lbpBlk;
		    LBP(img,lbpBlk,lbpRadius,blkX,blkY,blkSize);

		    /* Create histogram featurevector for LBP values of current block */
		    vector<int> hist;
		    featureVector(lbpBlk,hist);
		    for(int i=0; i<hist.size();i++) {
		    	featVectors.at<int>(blkIdx,i) = hist[i];
		    }

		    if(isTrainingsdata){
			    bool isRoad = (blkIdx>isRoadThreshold); // First/upper 60% of blocks is likely not road

			    /* Draw rectangle with block index */
			    int red=0,green=0;
			    isRoad? green=255 : red=255;
			    rectangle(dst
			        ,Point(blkX,blkY)
			        ,Point(blkX+blkSize,blkY+blkSize)
			        ,Scalar(0,green,red)
			    );

			    Point blkCenter(blkX+blkSize/2-5,blkY+blkSize/2+5);
			    strBldr.str(""); strBldr<<blkIdx;
			    putText(dst,strBldr.str(),blkCenter,FONT_HERSHEY_PLAIN,1,Scalar(0,green,red));

			    /* Write data to CSV-files */
			    fosLbl<<blkIdx<<","<<(isRoad? 1: -1)<<endl;

			    fosFv<<blkIdx<<",\"";
			    io::printVector(fosFv,hist);
			    fosFv<<"\""<<endl;
			}

		    /* Increment block coordinates */
		    blkX += blkSize+innerMargin;
		    if(blkX+blkSize+outerMargin > imWidth){
		        blkX = outerMargin;
		        blkY += blkSize+innerMargin;
		    }
		}
		fosLbl.close();
		fosFv.close();

		if(isTrainingsdata) io::saveImage(frameName+"_blocks",dst);
	}
};

#endif