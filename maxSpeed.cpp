#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/ml.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include "LineDetection/LineDetection.cpp"
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
#include <dirent.h>
#include <iomanip>
#include "Headers/svm.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int onderaan = 5;

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

void showMaxSpeed(vector<Mat> & masks, vector<Mat> & roads, vector<Mat> & frames, vector<vector<int> > roadRegions, vector<double> speeds, vector<bool> & lijndetectieBetrouwbaar) {
    int crash = 0;
    RNG rng(12345);
    int thresh = 255;
    for(int i = 0; i<masks.size(); i++) {
        Mat col = cv::Mat::ones(roads[i].rows, 1, roads[i].type());
        Mat rows = cv::Mat::ones(masks[i].rows-roads[i].rows, masks[i].cols, roads[i].type());
        //kolom en rijen toevoegen aan wegdetectie
        hconcat(roads[i], col, roads[i]);
        vconcat(rows,roads[i],roads[i]);

        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        
        //lijndetectie op wegdetectie
        Canny( roads[i], canny_output, thresh, thresh*2, 3 );

        //ROAD DETECTION indien lijndetectie niet voldoende betrouwbaar is
        //if(true){
        if(!lijndetectieBetrouwbaar[i]){
        //if(false){
            //rode outliers uithalen
            for(int j = 0; j < 10; j++){
                for(int z=39; z<roadRegions[i].size()-39; z++) {
                    if(roadRegions[i][z] == -1 && z%39 != 0 && z%39 != 38){
                        int aantalRodeBuren = 0;
                        //boven
                        if(roadRegions[i][z-39-1] != 1)
                            aantalRodeBuren++;
                        if(roadRegions[i][z-39] != 1)
                            aantalRodeBuren++;
                        if(roadRegions[i][z-39+1] != 1)
                            aantalRodeBuren++;
                        //L&R
                        if(roadRegions[i][z-1] != 1)
                            aantalRodeBuren++;
                        if(roadRegions[i][z+1] != 1)
                            aantalRodeBuren++;
                        //onder
                        if(roadRegions[i][z+39-1] != 1)
                            aantalRodeBuren++;
                        if(roadRegions[i][z+39] != 1)
                            aantalRodeBuren++;
                        if(roadRegions[i][z+39+1] != 1)
                            aantalRodeBuren++;
                        if(aantalRodeBuren < 4){
                            roadRegions[i][z] = 1;
                            //cout << "PUNT " << i%39 << "," << i/39 << " groen gemaakt."<< endl;
                        }
                    }
                }
            }

            for(int j=0; j<roadRegions[i].size()-39; j++) {
                int blkSize = 32;
                int blkX = (j%39)*blkSize;
                int blkY = (j/39)*blkSize;
                if(roadRegions[i][j]!=1){
                    rectangle(canny_output
                        ,Point(blkX,blkY)
                        ,Point(blkX+blkSize,blkY+blkSize)
                        ,Scalar(255,255,255)
                        );
                }
            }
        }
        
        //Randen van de weg vinden
        findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
        //Snijpunten tussen mask en wegranden toevoegen
        //Lijn tekenen van punt tot punt uit contours
        vector<Point> intersections;
        cv::Vec3b vec;
        for(int x = 0; x < contours.size(); x++){
            for(int y = 1; y < contours[x].size();y++){
                LineIterator it(roads[i], contours[x][y-1], contours[x][y], 8);
                for(int z = 0; z < it.count; z++, ++it){
                    it.pos();
                    vec = masks[i].at<cv::Vec3b>(it.pos().y,it.pos().x);
                    if(vec[0] != 0 && it.pos().y < masks[i].size().height - 5)
                        intersections.push_back(it.pos());
                }
            }
        }
        //Snijpunten tussen mask en framerand (L&R) toevoegen
        for(int y = 0; y < masks[i].size().height - 5; y++){
            //Links
            vec = masks[i].at<cv::Vec3b>(y,0);
            if(vec[0] != 0)
                intersections.push_back(Point(0,y));
            //Rechts
            vec = masks[i].at<cv::Vec3b>(y,masks[i].size().width-1);
            if(vec[0] != 0)
                intersections.push_back(Point(masks[i].size().width-1,y));
        }
        
        Mat dst;
        addWeighted( masks[i] , 1.0, frames[i], 1.0, 0.0, dst);
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 255,0,0 );
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( dst, contours, i, color, 2, 8, hierarchy, 0, Point(0,0) );
        }
        
        
        // Intersectiepunten
        Scalar color = Scalar( 0,255,0 );
        for(int j = 0; j < intersections.size(); j++){
            //cout << intersections[j] << endl;
            circle( dst, intersections[j], 2, color);
        }
        
        //Intersectiepunt dichtst bij auto
        float max;
        int laagsteSnelheid = 90;
        int laagsteSnelheidIndex = 0;
        for(int j = 0; j < intersections.size(); j++){
            vec = masks[i].at<cv::Vec3b>(intersections[j].y,intersections[j].x);
            max = vec[0];
            if(laagsteSnelheid > max){
                laagsteSnelheid = max;
                laagsteSnelheidIndex = j;
            }
        }
        
        if(intersections.size() != 0){
            circle( dst, intersections[laagsteSnelheidIndex], 2, Scalar( 0,0,255 ),2);
        }
        else{
            laagsteSnelheid = 90.0;
        }

        double minVal, maxVal;
        minMaxLoc(masks[i], &minVal, &maxVal);
        if(laagsteSnelheid > maxVal){
            laagsteSnelheid = maxVal;
        }
        
        if(laagsteSnelheid!= 90)
            laagsteSnelheid -=2;
        
        stringstream ss;
        ss << "Max speed (zelf): " << laagsteSnelheid << " km/u, gtdistances: " << speeds[i] << " km/u";
        if(laagsteSnelheid <= speeds[i])
            putText(dst, ss.str() , Point(10,30), 0, 1.0, Scalar( 0,255,0 ), 3);
        else{
            putText(dst, ss.str() , Point(10,30), 0, 1.0, Scalar( 0,0,255 ), 3);
            crash++;
        }
        
        //print result
        //cout << "FRAME " << i << ": " << laagsteSnelheid << " (" << speeds[i] << ") -> " << laagsteSnelheid-speeds[i] << endl;

        stringstream ss2;
        ss2 << setfill('0') << std::setw(5) << i << " " << laagsteSnelheid;
        cout << ss2.str() << endl;

        /// Show in a window
        /*namedWindow( "Max Speed", CV_WINDOW_AUTOSIZE );
        imshow( "Max Speed", dst );
        waitKey(0);*/
        stringstream ssOut;
        ssOut << "outputframes/frame" << i << ".png";
        imwrite(ssOut.str(),dst);
    }
    cout << "CRASHES: " << crash << endl;
}

vector<Mat> detectLines(vector<Mat> & masks, vector<Mat> & frames, vector<bool> & lijndetectieBetrouwbaar){
    LineDetection ld;
    int initialHoughVote = 150;
    int initialHoughVote2 = 150;
    int houghVote = initialHoughVote;
    int houghVote2 = initialHoughVote2;
    vector<Mat> lineContours;
    for(int i=0; i < frames.size(); i++) {
        bool drawLines = true; // draw detected lines on source image
        bool debugLinedetection = false; // wait after each frame and show all intermediate results
        bool betrouwbaar = false;
        Mat lineContour = ld.getLinesFromImage(frames[i], initialHoughVote, houghVote,initialHoughVote2, houghVote2, drawLines, debugLinedetection, betrouwbaar);
        lineContours.push_back(lineContour);
        lijndetectieBetrouwbaar.push_back(betrouwbaar);
        //namedWindow("LineContours");
        //imshow("LineContours",lineContour);
        //waitKey(0);
        //destroyAllWindows();
    }
    return lineContours;
}

vector<vector<int> > roadDetection(vector<Mat> & frames, string datasetFolder){
    vector<short> trainingDatasets;
    trainingDatasets.push_back(1);
    Mat initLabels,initTraining;

    io::readTrainingsdata(trainingDatasets,initLabels,initTraining);
    my_svm svm(initLabels,initTraining,true);
    
    // Show decision regions by the SVM
    vector<vector<int> > roadRegions;
    for(int i = 0; i < frames.size(); i++){
        vector<int> vec;

        LbpFeatureVector fv;
        Mat features;
        stringstream ss;
        ss << datasetFolder << "/frame" << setfill('0') << std::setw(5) << i << ".png";
        //cout << ss.str() << endl;
        fv.processFrame(ss.str(), frames[i], features);
    
        for(int j=0; j<features.rows; j++) {
            Mat_<float> row = Mat(features, Rect(0,j,features.cols,1));
            int response = svm.get_svm()->predict(row);
            vec.push_back(response);
        }
        roadRegions.push_back(vec);
    }
    return roadRegions;
}

int main(int argc, char** argv){
    if(argc <= 1) {
        cout << "add parameters: dataset folder" << endl;
        return 1;
    }
    
    vector<double> speeds;
    string line;
    std::stringstream ss;
    ss << argv[1] << "/gtdistances.txt";
    cout << ss.str() << endl;
    ifstream infile(ss.str().c_str());
    
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int imnr;
        double speed;
        iss >> imnr;
        iss >> speed;
        speeds.push_back(speed);
    }
    
    vector<Mat> masks = read_images(argv[1],"mask%05d.png");
    vector<Mat> frames = read_images(argv[1],"frame%05d.png");
    vector<bool> lijndetectieBetrouwbaar;
    vector<Mat> roads = detectLines(masks,frames,lijndetectieBetrouwbaar);
    cout << "lines done"<< endl;
    vector<vector<int> > roadRegions = roadDetection(frames,argv[1]);
    cout << "road done" << endl;

    /*for(int i = 0; i < roadRegions.size(); i++){
        cout << "FRAME " << i << endl;
        for(int j = 0; j < roadRegions[i].size(); j++){
            cout << roadRegions[i][j];
        }
    }*/
    
    system("exec rm -r outputframes/*");

    showMaxSpeed(masks,roads,frames,roadRegions,speeds,lijndetectieBetrouwbaar);
    
    //waitKey(0);
    destroyAllWindows();
    return 0;
}