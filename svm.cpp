#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include "Headers/svm.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    vector<short> trainingDatasets;
    trainingDatasets.push_back(1);
    Mat initLabels,initTraining;

	io::readTrainingsdata(trainingDatasets,initLabels,initTraining);
    my_svm svm(initLabels,initTraining);

    vector<short> extDatasets;
    extDatasets.push_back(1);
    Mat extLablesMat, extTrainingsMat;
    io::readTrainingsdata(extDatasets,extLablesMat,extTrainingsMat);

    double precision = svm.get_precision(extLablesMat,extTrainingsMat);
    cout << "precision of SVM: " << precision << "%" <<  endl;

    //use for visual testing
    svm.test(1,4,5,5);

	return 0;
}