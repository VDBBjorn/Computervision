#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <sstream>
#include <fstream>
#include "LineDetection/LineDetection.cpp"
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"

using namespace std;
using namespace cv;

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

void showMaxSpeed(vector<Mat> & masks, vector<Mat> & roads, vector<Mat> & frames, vector<double> speeds) {
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
            cout << j << ": " << intersections[j].y << "," << intersections[j].x << " -> " << max << ";" <<laagsteSnelheid << endl;
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
        
        stringstream ss;
        ss << "Max speed (zelf): " << laagsteSnelheid << " km/u, gtdistances: " << speeds[i] << " km/u";
        if(laagsteSnelheid <= speeds[i])
            putText(dst, ss.str() , Point(10,30), 0, 1.0, Scalar( 0,255,0 ), 1);
        else
            putText(dst, ss.str() , Point(10,30), 0, 1.0, Scalar( 0,0,255 ), 1);
        
        
        /// Show in a window
        namedWindow( "Max Speed", CV_WINDOW_AUTOSIZE );
        imshow( "Max Speed", dst );
        waitKey(0);
    }
}

vector<Mat> detectLines(vector<Mat> & masks, vector<Mat> & frames){
    LineDetection ld;
    int initialHoughVote = 150;
    int initialHoughVote2 = 150;
    int houghVote = initialHoughVote;
    int houghVote2 = initialHoughVote2;
    vector<Mat> lineContours;
    for(int i=0; i < frames.size(); i++) {
        bool drawLines = true; // draw detected lines on source image
        bool debugLinedetection = false; // wait after each frame and show all intermediate results
        Mat lineContour = ld.getLinesFromImage(frames[i], initialHoughVote, houghVote,initialHoughVote2, houghVote2, drawLines, debugLinedetection);
        lineContours.push_back(lineContour);
        //namedWindow("LineContours");
        //imshow("LineContours",lineContour);
        //waitKey(0);
        //destroyAllWindows();
    }
    return lineContours;
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
    ifstream infile(ss.str());
    
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
    
    vector<Mat> roads = detectLines(masks,frames);
    showMaxSpeed(masks,roads,frames, speeds);
    
    //waitKey(0);
    destroyAllWindows();
    return 0;
}