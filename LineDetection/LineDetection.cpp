#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include "LineFinder.hpp"

using namespace std;
using namespace cv;


class LineDetection {

public:
    Mat getLinesFromImage(Mat &image) {

        // parameters
        int cannyThreshold1 = 60;
        int cannyThreshold2 = 140;
        int threshThreshold = 128;
        int threshMaxval = 255;
        int houghPMinLineLength = 1;
        int houghPGap = 20;
        int houghPMinVote = 5;

        // set the ROI for the image
        Rect roi(0, image.cols / 4, image.cols - 1, image.rows - image.cols / 4);
        Mat imgROI = image(roi);

        // Canny algorithm
        Mat contours;
        Canny(imgROI, contours, cannyThreshold1, cannyThreshold2);
        Mat contoursInv;
        threshold(contours, contoursInv, threshThreshold, threshMaxval, THRESH_BINARY_INV);

        // Create LineFinder instance and do probabilistic Hough
        LineFinder ld;
        Mat houghP(imgROI.size(), CV_8U, Scalar(0));
        ld.setLineLengthAndGap(houghPMinLineLength, houghPGap);
        ld.setShift(0);
        ld.setMinVote(houghPMinVote);
        ld.findLines(contours);
        ld.drawDetectedLines(houghP);

        // Uncomment to test parameters
        /*
        namedWindow( "Source", CV_WINDOW_AUTOSIZE );
        imshow( "Source", image );
        namedWindow( "Lines", CV_WINDOW_AUTOSIZE );
        imshow( "Lines", houghP );
        waitKey(0);
        */

        return houghP;
    };

};