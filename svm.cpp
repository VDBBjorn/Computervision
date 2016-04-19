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

	for(int f=0; f<=100; f+=10) {

	    char number[36];
	    sprintf(number, "%05d", f);
	    string input = "output/frame"+string(number)+"_data.csv";
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

	cout << labelsMat << endl << trainingsMat << endl;

	// Train the SVM	    
    svm->train(trainingsMat, ROW_SAMPLE, labelsMat);
	//svm->trainAuto()

    // Show decision regions by the SVM
    Mat image = imread("Dataset/02/frame00000.png", CV_LOAD_IMAGE_COLOR);
    //Mat::zeros(720, 1280, CV_8UC3);
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,16) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response != 1)
                image.at<Vec3b>(i,j)  = blue;
        }
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    // Show support vectors
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }
    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

	return 0;
}