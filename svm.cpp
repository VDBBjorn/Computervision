#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include "Headers/svm.hpp"
#include <set>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    Mat initLabels,initTraining,testLabels, testTraining;

    set<int> tS;
    tS.insert(io::datasets[0]);
    tS.insert(io::datasets[2]);

    string frameName;
    char buffer[30];
    for(int dSIdx=0;dSIdx<sizeof(io::datasets)/sizeof(int);dSIdx++){
    int dataset = io::datasets[dSIdx];
    for(int fIdx=0;fIdx<io::frameStopIdx;fIdx+=io::frameInterval){
        io::buildFrameName(buffer,frameName,dataset,fIdx,io::innerMargin,io::blkSize,io::includeMarks);
        bool isTrainingSet = tS.find(dataset)!=tS.end();
        if(isTrainingSet)
            io::readTrainingsdataOutput(frameName,initLabels,initTraining,io::useLBP,io::useColor);
        else
            io::readTrainingsdataOutput(frameName,testLabels,testTraining,io::useLBP,io::useColor);
    }
    }

    my_svm svm(initLabels,initTraining,io::trainAuto);
    svm.calculateScores(testLabels,testTraining);

    //use for visual testing
    svm.test(1,1,10,5);

	return 0;
}