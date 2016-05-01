#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include "Headers/svm.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	vector<short> labels;
	vector<vector<int> > featureVectors;

    my_svm svm(labels,featureVectors);

    double precision = svm.get_precision();
    cout << "precision of SVM: " << precision << "%" <<  endl;

    //use for visual testing
    svm.test(1,4,5,5);

	return 0;
}