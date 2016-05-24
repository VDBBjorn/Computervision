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
    for(int dSIdx=0;dSIdx<sizeof(io::datasets)/sizeof(int);dSIdx++){
    int dataset = io::datasets[dSIdx];
    for(int fIdx=0;fIdx<io::frameStopIdx;fIdx+=io::frameInterval){
        io::buildFrameName(buffer,frameName,dataset,fIdx,io::innerMargin,io::blkSize,io::includeMarks);
        io::readTrainingsdataOutput(frameName,initLabels,initTraining);
        io::readTrainingsdataOutput(frameName,testLabels,testTraining);
    }
    }

    // io::readTrainingsdata(trainingDatasets,initLabels,initTraining);
    // io::readTrainingsdata(testDatasets,testLabels,testTraining);

    my_svm svm(initLabels,initTraining,io::trainAuto);
    svm.calculateScores(testLabels,testTraining);

    //use for visual testing
    svm.test(1,4,10,5);

	return 0;
}