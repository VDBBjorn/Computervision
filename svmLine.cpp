#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;

vector<Mat> read_images(String folder, String regex) {
    VideoCapture cap(folder+"/"+regex);
    vector<Mat> matrices;
    while( cap.isOpened() )
    {
        Mat img;
        if(!cap.read(img)) {
            return matrices;
        }
        matrices.push_back(img);
    }
    return matrices;
}

int main(int argc, char** argv){
	if(argc <= 1) {
		cout << "add parameters: dataset folder" << endl;
		return 1;
	}	

	//vector<Mat> frames;
	//frames = read_images(argv[1],"frame%05d.png");
    
    FILE *myFile;
    myFile = fopen("svmFrame26.txt", "r");
    
    vector<int> resVector;
    int res;
    int i = 0;
    
    while(fscanf(myFile, "%d,", &res ) != EOF){
        resVector.push_back(res);
        //cout << resVector[i] ;
        i++;
    }
    //cout << endl;
    for(int i = resVector.size()-39; i < resVector.size(); i++){
        resVector[i] = 1;
    }
    
    string fnFrame = "Dataset/01/frame00026.png";
    Mat image = imread(fnFrame, CV_LOAD_IMAGE_COLOR);
    
    for(int i=0; i<resVector.size(); i++) {
        int red=0,green=0;
        int blkSize = 32;
        int blkX = (i%39)*blkSize;
        int blkY = (i/39)*blkSize;
        (resVector[i]==1? green=255 : red=255);
        rectangle(image
                  ,Point(blkX,blkY)
                  ,Point(blkX+blkSize,blkY+blkSize)
                  ,Scalar(0,green,red)
                  );
    }
	
    imshow("output", image);
    waitKey(0);
    
    for(int j = 0; j < 10; j++){
    //rode outliers uithalen
    for(int i=39; i<resVector.size()-39; i++) {
        if(resVector[i] == -1 && i%39 != 0 && i%39 != 38){
            int aantalRodeBuren = 0;
            //boven
            if(resVector[i-39-1] != 1)
                aantalRodeBuren++;
            if(resVector[i-39] != 1)
                aantalRodeBuren++;
            if(resVector[i-39+1] != 1)
                aantalRodeBuren++;
            //L&R
            if(resVector[i-1] != 1)
                aantalRodeBuren++;
            if(resVector[i+1] != 1)
                aantalRodeBuren++;
            //onder
            if(resVector[i+39-1] != 1)
                aantalRodeBuren++;
            if(resVector[i+39] != 1)
                aantalRodeBuren++;
            if(resVector[i+39+1] != 1)
                aantalRodeBuren++;
            if(aantalRodeBuren < 4){
                resVector[i] = 1;
                //cout << "PUNT " << i%39 << "," << i/39 << " groen gemaakt."<< endl;
            }
        }
    }
    
    image = imread(fnFrame, CV_LOAD_IMAGE_COLOR);
    
    for(int i=0; i<resVector.size(); i++) {
        int red=0,green=0;
        int blkSize = 32;
        int blkX = (i%39)*blkSize;
        int blkY = (i/39)*blkSize;
        (resVector[i]==1? green=255 : red=255);
        rectangle(image
                  ,Point(blkX,blkY)
                  ,Point(blkX+blkSize,blkY+blkSize)
                  ,Scalar(0,green,red)
                  );
    }
    
    imshow("output", image);
        
    }
    waitKey(0);
    destroyAllWindows();
	return 0;
}																																																																																							

