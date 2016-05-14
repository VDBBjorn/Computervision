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
#include "lbpfeaturevector.hpp"

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
	double recall;
	double accuracy;
	double true_negative;
	double F;

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
		    svm->trainAuto(trainData_ptr,10,SVM::getDefaultGrid(SVM::C),SVM::getDefaultGrid(SVM::GAMMA),ParamGrid(),ParamGrid(),ParamGrid(),ParamGrid(),true);
		}else{
		    /* Hardcoded params */
		    svm->setC(io::C);
		    svm->setCoef0(0);
		    svm->setDegree(0);
		    svm->setGamma(io::gamma);
		    svm->setNu(0);
		    svm->setP(0);
		    svm->train(trainData_ptr);
		}
		cout << "SVM trained" << endl;
		confusion = Mat::zeros(2,2, CV_32S);
		precision = 0.0;
		recall = 0.0;
		accuracy = 0.0;
		true_negative = 0.0;
		F = 0.0;

	    printParams(cout);
	}

	Ptr<SVM> get_svm();
	void calculateScores(Mat&, Mat&);
	double get_precision();
	double get_recall();
	double get_accuracy();
	double get_true_negative();
	double get_F();
	Mat* get_confusion_matrix();
	void test(int, int, int, int);
	void printParams(ostream&);

};

Ptr<SVM> my_svm::get_svm() {
	return svm;
}

/**
 * Returns precision of the SVM on the given dataset after training
**/
double my_svm::get_precision(){
	return precision;
}

double my_svm::get_recall(){
	return recall;
}

double my_svm::get_accuracy(){
	return accuracy;
}

double my_svm::get_true_negative(){
	return true_negative;
}

double my_svm::get_F(){
	return F;
}

Mat* my_svm::get_confusion_matrix(){
	return &confusion;
}

void my_svm::calculateScores(Mat& extLabels, Mat& extTrainingsMat) {
    cout<<"Start calculateScores"<<endl;
	if(precision == 0 || recall == 0 || accuracy == 0 || true_negative == 0) {
		for(int i=0; i<extTrainingsMat.rows;i++) {
		    Mat value = extTrainingsMat.row(i);
		    int label = extLabels.at<int>(i,0);
		    int predicted = svm->predict(value);
		    if(label==1 && predicted == 1) {	//TP
		        confusion.at<int>(0,0)++;
		    }
		    else if (label==-1 && predicted==-1) {	//TN
		        confusion.at<int>(1,1)++;
		    }
		    else {
		        confusion.at<int>(label==-1 ? 0,1 : 1,0)++;
		    }
		}
		accuracy = ((double)(confusion.at<int>(0,0)+confusion.at<int>(1,1)))/((double)extTrainingsMat.rows)*100;
		precision = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(0,1))*100);
		recall = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(1,1)))*100;
		if(confusion.at<int>(1,0) == 0) true_negative = 0;
		else true_negative = (double)(confusion.at<int>(1,0)/(confusion.at<int>(1,0)+confusion.at<int>(0,1))*100);	
		F = 2*(precision*recall)/(precision+recall);
	}
    cout<<"Confusion matrix: "<<endl<<confusion<<endl;
    cout<<"Precision: "<<precision<<"%"<<endl;
    cout<<"Recall: "<<recall<<"%"<<endl;
    cout<<"Accuracy: "<<accuracy<<"%"<<endl;
    cout<<"True Negative: "<<true_negative<<"%"<<endl;    
    cout<<"F: "<<F<<"%"<<endl;
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

            int blkSize = fv.getBlkSize(), imWidth = image.cols, imHeight = image.rows;
			int numBlksWidth = (imWidth-2*fv.getOuterMargin()-blkSize)/(fv.getInnerMargin()+blkSize) + 1;

            for(int i=0; i<features.rows; i++) {
                Mat_<float> row = Mat(features, Rect(0,i,features.cols,1));
                int response = svm->predict(row);
                int red=0,green=0;
                int blkX = (i%numBlksWidth)*blkSize;
                int blkY = (i/numBlksWidth)*blkSize;
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