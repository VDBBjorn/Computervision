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
	LbpFeatureVector(): outerMargin(1),innerMargin(0),blkSize(32),lbpRadius(1),histBins(16){}

	LbpFeatureVector(int _outerMargin,int _innerMargin,int _blkSize,int _lbpRadius,int _histBins):outerMargin(_outerMargin),innerMargin(_innerMargin),blkSize(_blkSize),lbpRadius(_lbpRadius),histBins(_histBins){}

	/* Calculate LBP values of pixels in src to dst, with radius r.
	* Only calculate withing given window, measured by source coordinates (x,y) and winSize.
	* Based on lbp code from https://github.com/bytefish/opencv/tree/master/lbp
	*/
	template <typename _Tp>
	void _LBP(const Mat& src, Mat& dst, int r, int x, int y, int winSize) {
	    if( x<r || y<r || x+winSize+r>src.cols || y+winSize+r>src.rows){
	        cerr << "Error in _LBP : source coordinates and window size make radius exceed image size" << endl;
	        throw;
	    }
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
	// template <typename _Tp>
	// void _LBP(const Mat& src, Mat& dst, int r, int x, int y, int winSize) {
	//     if( x<r || y<r || x+winSize+r>src.cols || y+winSize+r>src.rows){
	//         cerr << "Error in LBP_ : source coordinates and window size make radius exceed image size" << endl;
	//         throw;
	//     }
	//     dst = Mat::zeros(winSize, winSize, CV_8UC(src.channels()));
	//     for(int i=y;i<y+winSize;i++) {
	//         for(int j=x;j<x+winSize;j++) {
	//             _Tp center = src.at<_Tp>(i,j);
	//             unsigned char code = 0;
	//             code |= (src.at<_Tp>(i-r,j-r) > center) << 7;
	//             code |= (src.at<_Tp>(i-r,j) > center) << 6;
	//             code |= (src.at<_Tp>(i-r,j+r) > center) << 5;
	//             code |= (src.at<_Tp>(i,j+r) > center) << 4;
	//             code |= (src.at<_Tp>(i+r,j+r) > center) << 3;
	//             code |= (src.at<_Tp>(i+r,j) > center) << 2;
	//             code |= (src.at<_Tp>(i+r,j-r) > center) << 1;
	//             code |= (src.at<_Tp>(i,j-r) > center) << 0;
	//             dst.at<unsigned char>(i-y,j-x) = code;
	//         }
	//     }
	// }

	/*
	* Creates histogram featurevector for src
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
		const int UCHAR = CV_8UC(src.channels());
		if(src.type()==UCHAR){
			_LBP<unsigned char>(src, dst, radius, x, y, winSize);
		}else{

		    switch(src.type()) {
		        case CV_8SC1: _LBP<char>(src, dst, radius, x, y, winSize); break;
		        // case TYPE_CHAR: _LBP<unsigned char>(src, dst, radius, x, y, winSize); break;
		        case CV_16SC1: _LBP<short>(src, dst, radius, x, y, winSize); break;
		        case CV_16UC1: _LBP<unsigned short>(src, dst, radius, x, y, winSize); break;
		        case CV_32SC1: _LBP<int>(src, dst, radius, x, y, winSize); break;
		        case CV_32FC1: _LBP<float>(src, dst, radius, x, y, winSize); break;
		        case CV_64FC1: _LBP<double>(src, dst, radius, x, y, winSize); break;
		        default: cerr<<"Error in LBP : convert src to grayscale with valid type"<<endl;throw;
		    }
		}
	}

	/* Wrapper function for histogram_ */
	/* Based on code from https://github.com/bytefish/opencv/tree/master/lbp */
	void histogram(const Mat& src, vector<int>& hist, int numBins) {
		const int UCHAR = CV_8UC(src.channels());
		if(src.type()==UCHAR){
			_histogram<unsigned char>(src, hist, numBins);
		}else{

		    switch(src.type()) {
		        case CV_8SC1: _histogram<char>(src, hist, numBins); break;
		        case CV_8UC1: _histogram<unsigned char>(src, hist, numBins); break;
		        case CV_16SC1: _histogram<short>(src, hist, numBins); break;
		        case CV_16UC1: _histogram<unsigned short>(src, hist, numBins); break;
		        case CV_32SC1: _histogram<int>(src, hist, numBins); break;
		        default: cerr<<"Error in histogram : convert src to grayscale with valid type"<<endl;throw;
		    }
		}
	}

	void featureVector(const Mat& src,vector<int>& fv){
		vector<Mat> srcChnls(src.channels());
		split(src,srcChnls);
		for(int i=0;i<srcChnls.size();i++){
			vector<int> hist;
			histogram(srcChnls[i],hist,histBins);
			fv.insert(fv.end(),hist.begin(),hist.end());
		}
	}

	void processFrame(string fnFrame, Mat& img, Mat& featVectors){
    	ostringstream strBldr; // Stringbuilder

    	cout<<"Start processFrame "<<fnFrame<<endl;
    	/* Initial configuration */
		blkX = outerMargin;
		blkY = outerMargin;
		if(outerMargin<lbpRadius) outerMargin = lbpRadius;

		/*    Process frame   */
		string frameName = fnFrame.substr(0,fnFrame.size()-4);
		
		if(! img.data){
		    cerr <<  "Could not open or find frame " << frameName << std::endl ;
		    throw;
		}
		int imWidth=img.cols;
		int imHeight=img.rows;

		//vector<vector<int> > featVectors; // TODO For passing to trained SVM
		io::checkDir(io::dirOutput);

		/* Convert to grayscale for processing */
		Mat imGray;
		// cvtColor(img,imGray,CV_BGR2GRAY);
		imGray = img;

		/* Open new CSV-file for frame data in output directory */
		string fn = io::dirOutput+frameName+"_data.csv";
		ofstream fos(fn.c_str(),ios::out);

		int numBlksWidth = (imWidth-2*outerMargin-blkSize)/(innerMargin+blkSize) + 1;
		int numBlksHeight = (imHeight-2*outerMargin-blkSize)/(innerMargin+blkSize) + 1;
		int totalBlocks = numBlksWidth*numBlksHeight - 1; // -1 Accounts for 0-indexing
		int isRoadThreshold = totalBlocks*0.6;


		featVectors = Mat(totalBlocks+1,histBins*img.channels(),CV_32SC1);	

		/* Process blocks in frame */
		int blkIdx = -1;
		while(blkX+blkSize+outerMargin < imWidth && blkY+blkSize+outerMargin < imHeight){
		    blkIdx++;
		    bool isRoad = (blkIdx>isRoadThreshold); // First/upper 60% of blocks is likely not road

		    /* Draw rectangle with block index */
		    int red=0,green=0;
		    isRoad? green=255 : red=255;
		    rectangle(img
		        ,Point(blkX,blkY)
		        ,Point(blkX+blkSize,blkY+blkSize)
		        ,Scalar(0,green,red)
		    );

		    Point blkCenter(blkX+blkSize/2-5,blkY+blkSize/2+5);
		    strBldr.str(""); strBldr<<blkIdx;
		    putText(img,strBldr.str(),blkCenter,FONT_HERSHEY_PLAIN,1,Scalar(0,green,red));

		    /* Calculate LBP values for current block */
		    Mat lbpBlk;
		    LBP(imGray,lbpBlk,lbpRadius,blkX,blkY,blkSize);

		    /* Create histogram featurevector for LBP values of current block */
		    vector<int> hist;
		    featureVector(lbpBlk,hist);
		    for(int i=0; i<hist.size();i++) {
		    	featVectors.at<int>(blkIdx,i) = hist[i];
		    }

		    /* Write data to CSV-file */
		    fos<<blkIdx<<",\"";
		    io::printVector(fos,hist);
		    fos<<"\","<< (isRoad? 1: -1);
		    fos<<endl;

		    /* Increment block coordinates */
		    blkX += blkSize+innerMargin;
		    if(blkX+blkSize+outerMargin > imWidth){
		        blkX = outerMargin;
		        blkY += blkSize+innerMargin;
		    }
		}
		fos.close();

		io::saveImage(frameName+"_testblocks",img);

	    // io::showImage(fnFrame,img,false);
	    // waitKey(0);
	    // destroyWindow(fnFrame);
	}
};

#endif