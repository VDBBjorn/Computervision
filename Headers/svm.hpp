#ifndef SVM_
#define SVM_

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

class my_svm {	
private:
	Ptr<SVM> svm;
	// Mat labelsMat;
	// Mat trainingsMat;
	Mat confusion;
	double precision;

public:
	my_svm(Mat& labelsMat, Mat& trainingsMat, bool trainAuto=false) {
		svm = SVM::create();
	    svm->setType(SVM::C_SVC);
	    svm->setKernel(SVM::RBF);
	    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	    Ptr<TrainData> trainData_ptr = TrainData::create(trainingsMat, ROW_SAMPLE , labelsMat);
		cout << "training the SVM... "<<(trainAuto? "(automatically, this could take a while...)" : "") << endl;
	    if(trainAuto){
		    /* Automatic training */
		    svm->trainAuto(trainData_ptr);
		}else{
		    /* Hardcoded params */
		    svm->setC(0.1);
		    svm->setCoef0(0);
		    svm->setDegree(0);
		    svm->setGamma(0.00015);
		    svm->setNu(0);
		    svm->setP(0);
		    svm->train(trainData_ptr);
		}
		cout << "SVM trained" << endl;
	    printParams(cout);
		confusion = Mat::zeros(2,2, CV_32S);
		precision = 0.0;
	}

	Ptr<SVM> get_svm();
	double get_precision(Mat&, Mat&);
	void test(int, int, int, int);
	void printParams(ostream&);

};

Ptr<SVM> my_svm::get_svm() {
	return svm;
}

/**
 * Returns precision of the SVM on the given dataset after training
**/
double my_svm::get_precision(Mat& extLabels, Mat& extTrainingsMat) {
	if(precision == 0) {
		for(int i=0; i<extTrainingsMat.rows;i++) {
		    Mat value = extTrainingsMat.row(i);
		    int label = extLabels.at<int>(i,0);
		    int predicted = svm->predict(value);
		    if(label==1 && predicted == 1) {
		        confusion.at<int>(0,0)++;
		    }
		    else if (label==-1 && predicted==-1) {
		        confusion.at<int>(1,1)++;
		    }
		    else {
		        confusion.at<int>(label==-1 ? 0,1 : 1,0)++;
		    }
		}
		precision = ((double)(confusion.at<int>(0,0)+confusion.at<int>(1,1)))/((double)extTrainingsMat.rows)*100;
	}
	cout<<"Confusion matrix: "<<endl;
	cout<<confusion<<endl;
	return precision;
}

/**
 * param min: first dataset to test (default = 1)
 * param max: last dataset to test (default = 4)
 * param skip: number of frames to skip in between the frames (default = 5)
 * param number: number of frames to test on (default = 5)
**/
void my_svm::test(int min = 1, int max = 4, int skip = 5, int number = 5) {
	int keyboard = -1;
    for(int ds=min; ds<=max; ds++) {
        char dsnumber[36];
        sprintf(dsnumber, "%02d", ds);
        for(int f=0; f<=(number*skip); f+=5) {
            char number[36];
            sprintf(number, "%05d", f);
            // Show decision regions by the SVM
            string fnFrame = "Dataset/"+string(dsnumber)+"/frame"+string(number)+".png";
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
            imshow("output", image); // show it to the user
			
			if((keyboard=waitKey(0)) == io::KEY_ESCAPE)
				break;
        }
    }
    destroyAllWindows();
}

void my_svm::printParams(ostream& os){
	os<<"--- Params ---"<<endl;
	os<<"C:\t\t"<<				svm->getC()<<endl;
	os<<"Class weights:\t"<<	svm->getClassWeights()<<endl;
	os<<"Coef0:\t\t"<<			svm->getCoef0()<<endl;
	os<<"Degree:\t\t"<<			svm->getDegree()<<endl;
	os<<"Gamma:\t\t"<<			svm->getGamma()<<endl;
	os<<"Nu:\t\t"<<				svm->getNu()<<endl;
	os<<"P:\t\t"<<				svm->getP()<<endl;
}
#endif