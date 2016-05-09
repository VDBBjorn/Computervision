#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include "Headers/svm.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    vector<short> trainingDatasets,testDatasets;
    trainingDatasets.push_back(1);
    testDatasets.push_back(1);
    Mat initLabels,initTraining,testLabels, testTraining;

    string frameName;
    char buffer[30];
    int dataset(1),innerMargin(64),blkSize(16);
    for(int f=0; f<=200; f+=10) {
        sprintf(buffer,"%02dframe%05d_%03d_%02d",dataset,f,innerMargin,blkSize);
        frameName = string(buffer);
        io::readTrainingsdataOutput(frameName,initLabels,initTraining);
        io::readTrainingsdataOutput(frameName,testLabels,testTraining);
    }

    // io::readTrainingsdata(trainingDatasets,initLabels,initTraining);
    // io::readTrainingsdata(testDatasets,testLabels,testTraining);

    my_svm svm(initLabels,initTraining,true);
    svm.calculateScores(testLabels,testTraining);

    //use for visual testing
    svm.test(1,4,5,5);

	return 0;
}