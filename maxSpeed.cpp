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
#include <dirent.h>
#include <iomanip>

#include "Headers/LineDetection.hpp"
#include "Headers/lbpfeaturevector.hpp"
#include "Headers/io.hpp"
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

/**
 * Calculate for a given point whether it is detected as road or not.
 * For safety reasons, a point is only considered as road when the above, left and right blocks are classified as road
 */
bool isRoad(vector<int> & roadRegions, int blksInWidth, int frameRows, int frameCols, int x, int y) {
    int j = (y / io::blkSize) * blksInWidth + (x / io::blkSize);

    // most left and right column are never considered as road, in order to limit the speed when
    // reaching the edge of the frame (i.e. necessary on the roundabout in Dataset 01)
    if(x < io::blkSize || x >= frameCols - io::blkSize) return false;

    // the lowest row is always considered as road
    if(y >= frameRows - io::blkSize) return true;

    if(
        roadRegions[j] == 1 &&
        // above
        roadRegions[j-blksInWidth-1] == 1 &&
        roadRegions[j-blksInWidth] == 1 &&
        roadRegions[j-blksInWidth+1] == 1 &&

        // left & right
        roadRegions[j-1] == 1 &&
        roadRegions[j+1] == 1
    ) {
        return true;
    }
}

void showMaxSpeed(vector<Mat> & masks, vector<Mat> & roads, vector<Mat> & frames, vector<vector<int> > roadRegions, vector<double> speeds, string dirOutputFrames, string& results) {
    int crash = 0;

    int thresh = 255;
    stringstream ssOut;
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

        int blksInWidth = (canny_output.cols-2*io::lbpRadius)/io::blkSize; // (width-2*outerMargin)/blkSize - 1, -1 to compensate for 0-indexing

        // The lowest blocklines are always considered as road and are included when calculating the outliers
        for(int j=(frames[i].rows - io::blkSize * 2) / io::blkSize * blksInWidth; j < roadRegions[i].size(); j++) {
            roadRegions[i][j] = 1;
        }


        //rode outliers uithalen
        bool changedOutlier = true;
        while(changedOutlier) {
            changedOutlier = false;
            for (int z = blksInWidth; z < roadRegions[i].size() - blksInWidth; z++) {
                if (roadRegions[i][z] == -1 && z % blksInWidth != 0 && z % blksInWidth != blksInWidth - 1) {
                    int aantalRodeBuren = 0;
                    //boven
                    if (roadRegions[i][z - blksInWidth - 1] != 1)
                        aantalRodeBuren++;
                    if (roadRegions[i][z - blksInWidth] != 1)
                        aantalRodeBuren++;
                    if (roadRegions[i][z - blksInWidth + 1] != 1)
                        aantalRodeBuren++;
                    //L&R
                    if (roadRegions[i][z - 1] != 1)
                        aantalRodeBuren++;
                    if (roadRegions[i][z + 1] != 1)
                        aantalRodeBuren++;
                    //onder
                    if (roadRegions[i][z + blksInWidth - 1] != 1)
                        aantalRodeBuren++;
                    if (roadRegions[i][z + blksInWidth] != 1)
                        aantalRodeBuren++;
                    if (roadRegions[i][z + blksInWidth + 1] != 1)
                        aantalRodeBuren++;

                    if (aantalRodeBuren < 4) {
                        roadRegions[i][z] = 1;
                        changedOutlier = true;
                        //cout << "PUNT " << z%blksInWidth << "," << z/blksInWidth << " groen gemaakt."<< endl;
                    }
                }
            }
        }

        // Add non-road rectangles to the canny output
        // --> Disabled
        /*
        for(int j=0; j<roadRegions[i].size()-blksInWidth; j++) {
            int blkX = (j%blksInWidth)*io::blkSize;
            int blkY = (j/blksInWidth)*io::blkSize;
            // cout<<roadRegions[i].size()<<","<<j<<","<<blkX<<","<<blkY<<endl;
            if(roadRegions[i][j]!=1){
                rectangle(canny_output
                    ,Point(blkX,blkY)
                    ,Point(blkX+io::blkSize,blkY+io::blkSize)
                    ,Scalar(255,255,255)
                    );
            }
        }*/

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

                    // Only add line segments when it is clear that it isn't on the road
                    if(!isRoad(roadRegions[i], blksInWidth, frames[i].rows, frames[i].cols, it.pos().x, it.pos().y)) {
                        if (vec[0] != 0 && it.pos().y < masks[i].size().height - 5)
                            intersections.push_back(it.pos());
                    }
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

        // Visualize road blocks on the output frame
        for(int x=0; x < canny_output.cols; x+=io::blkSize) {
            for(int y=0; y < canny_output.rows; y+=io::blkSize) {
                if(isRoad(roadRegions[i], blksInWidth, frames[i].rows, frames[i].cols, x, y)) {
                    rectangle(dst
                            ,Point(x,y)
                            ,Point(x+io::blkSize,y+io::blkSize)
                            ,Scalar(255,0,0)
                    );

                }
            }
        }

        // Visualize detected lines
        for( int j = 0; j < contours.size(); j++ )
        {
            Scalar color = Scalar( 255,0,0 );
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( dst, contours, j, color, 2, 8, hierarchy, 0, Point(0,0) );
        }
        
        
        // Intersectiepunten (= groene zone)
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
            laagsteSnelheid -=1;
        
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

        ssOut << setfill('0') << std::setw(5) << i << " " << laagsteSnelheid <<endl;

        /// Show in a window
        /*namedWindow( "Max Speed", CV_WINDOW_AUTOSIZE );
        imshow( "Max Speed", dst );
        waitKey(0);*/
        io::checkDir(dirOutputFrames);
        stringstream ssFilename;
        ssFilename << dirOutputFrames << "/frame" << setfill('0') << std::setw(5) << i << ".png";
        imwrite(ssFilename.str(),dst);
    }
    ssOut << "CRASHES: " << crash << endl;

    results = ssOut.str();
    cout<<results<<endl;
}

vector<Mat> detectLines(vector<Mat> & masks, vector<Mat> & frames){
    LineDetection ld;
    vector<Mat> lineContours;
    for(int i=0; i < frames.size(); i++) {
        Mat lineContour = ld.getLinesFromImage(frames[i]);
        lineContours.push_back(lineContour);
    }
    return lineContours;
}

vector<vector<int> > roadDetection(vector<Mat> & frames){
    Mat initLabels,initTrainingsdata;

    char buffer[30];
    string frameName;

    for(int dSIdx=0;dSIdx<sizeof(io::datasets)/sizeof(int);dSIdx++){
    int dataset = io::datasets[dSIdx];
    for(int fIdx=0;fIdx<io::frameStopIdx;fIdx+=io::frameInterval){
        io::buildFrameName(buffer,frameName,dataset,fIdx,io::innerMargin,io::blkSize,io::includeMarks);
        io::readTrainingsdataOutput(frameName,initLabels,initTrainingsdata,io::useLBP,io::useColor);
    }
    }

    my_svm svm(initLabels,initTrainingsdata,io::trainAuto);
    
    // Show decision regions by the SVM
    vector<vector<int> > roadRegions;

    LbpFeatureVector fv;
    Mat features;
    for(int i = 0; i < frames.size(); i++){
    // for(int i = 0; i < 1; i++){
        vector<int> vec;
        io::buildFrameName(buffer,frameName,0,i,0,fv.getBlkSize(),io::includeMarks);
        fv.processFrame(frameName, frames[i], features);

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
        cout << "add parameters: datasetfolder" << endl;
        return 1;
    }
    string dirDataset(argv[1]);
    string results;
    
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
    vector<Mat> masks = read_images(dirDataset,"mask%05d.png");
    vector<Mat> frames = read_images(dirDataset,"frame%05d.png");

    vector<Mat> roads = detectLines(masks,frames);
    cout << "lines done"<< endl;
    vector<vector<int> > roadRegions = roadDetection(frames);
    cout << "road done" << endl;

    /*for(int i = 0; i < roadRegions.size(); i++){
        cout << "FRAME " << i << endl;
        for(int j = 0; j < roadRegions[i].size(); j++){
            cout << roadRegions[i][j];
        }
    }*/
    
    string dirOutputFrames = "outputframes/"+dirDataset;
    io::checkDir(dirOutputFrames);
    system(string("exec rm -r "+dirOutputFrames+"/*").c_str());

    showMaxSpeed(masks,roads,frames,roadRegions,speeds,dirOutputFrames,results);

    ofstream osResults;
    osResults.open(string(dirDataset+"/results.txt").c_str(),ios::out);
    osResults<<results<<endl;
    osResults.close();
    
    //waitKey(0);
    destroyAllWindows();
    return 0;
}