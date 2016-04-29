#include <string>
#include <stdio.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include <dirent.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main (int argc, char** argv){
	Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    //svm->setDegree(2);
    // svm->setGamma(3); 
    //svm->setNu(0.4);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	vector<short> labels;
	vector<vector<int> > featureVectors;

	io::readTrainingsdata(labels,featureVectors);

	// Set up training data and labels
 	Mat labelsMat = Mat::zeros(labels.size(), 1, CV_32SC1);
 	for(int i=0; i<labels.size();i++) {
		labelsMat.at<int>(i,0) = labels[i];
	}

 	Mat trainingsMat = Mat::zeros(featureVectors.size(), featureVectors[0].size(), CV_32FC1);
	for(int i = 0; i<featureVectors.size(); i++) {
		for(int j=0; j<featureVectors[i].size(); j++) {
			trainingsMat.at<float>(i,j) = featureVectors[i][j];
		}
	}

	cout << labelsMat << endl << trainingsMat << endl;

	// Train the SVM	    
    //svm->train(trainingsMat, ROW_SAMPLE, labelsMat);
    Ptr<TrainData> trainData_ptr = TrainData::create(trainingsMat, ROW_SAMPLE , labelsMat);
    svm->trainAuto(trainData_ptr);
    cout << "SVM trained" << endl;


    // Show decision regions by the SVM
    string fnFrame = "Dataset/01/frame00100.png";
    Mat image = imread(fnFrame, CV_LOAD_IMAGE_COLOR);
    LbpFeatureVector fv;
    Mat features;
    fv.processFrame(fnFrame, image, features);

    // cout << features << endl;

    for(int i=0; i<features.rows; i++) {
    	Mat_<float> row = Mat(features, Rect(0,i,features.cols,1));
    	int response = svm->predict(row);
    	int red=0,green=0;
    	int blkSize = 32;
    	int blkX = (i%39)*blkSize;
    	int blkY = (i/39)*blkSize;
	    (response==1? green=255 : red=255);
	    rectangle(image
	        ,Point(blkX,blkY)
	        ,Point(blkX+blkSize,blkY+blkSize)
	        ,Scalar(0,green,red)
	    );
    }
    imwrite("result.png", image);        // save the image
    imshow("output", image); // show it to the user
    waitKey(0);

	return 0;
}