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
    svm->setDegree(2);
    svm->setGamma(3); 
    svm->setKernel(SVM::POLY);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	vector<vector<int> > matrices;
	vector<int> labels;

	for(int f=0; f<=0; f+=10) {

	    char number[36];
	    sprintf(number, "%05d", f);
	    string input = "output/frame"+string(number)+"_data-correct.csv";
	    cout << input << endl;; 
	 	ifstream file(input.c_str());
		string value;
		int i=0;
		while(getline(file,value)) {
			vector<string> line;
			stringstream ss(value);
			string item;
			while(getline(ss,item,',')) {
				line.push_back(item);
			}
			stringstream sst(string(line[1],1,line[1].length()-2));
			vector<int> mat_values;
			while(getline(sst,item,';')) {
				mat_values.push_back(atoi(item.c_str()));
			}
			matrices.push_back(mat_values);
			labels.push_back(atoi(line[2].c_str()));
		}
	}

	// Set up training data and labels
 	Mat labelsMat = Mat::zeros(labels.size(), 1, CV_32SC1);
 	for(int i=0; i<labels.size();i++) {
		labelsMat.at<int>(i,0) = labels[i];
	}

 	Mat trainingsMat = Mat::zeros(matrices.size(), matrices[0].size(), CV_32FC1);
	for(int i = 0; i<matrices.size(); i++) {
		for(int j=0; j<matrices[i].size(); j++) {
			trainingsMat.at<float>(i,j) = matrices[i][j];
		}
	}

	// cout << labelsMat << endl << trainingsMat << endl;

	// Train the SVM	    
    svm->train(trainingsMat, ROW_SAMPLE, labelsMat);
	//svm->trainAuto()
    cout << "svm trained" << endl;
    // Show decision regions by the SVM
    Mat image = imread("Dataset/01/frame00000.png", CV_LOAD_IMAGE_COLOR);
    //Mat::zeros(720, 1280, CV_8UC3);
    //Vec3b green(0,255,0), blue (255,0,0);
    LbpFeatureVector fv;
    Mat features;
    fv.processFrame("test-data", image, features);
    image = imread("Dataset/01/frame00000.png", CV_LOAD_IMAGE_COLOR);
    cout << "--------------------------------" << endl;
    cout << features << endl;

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