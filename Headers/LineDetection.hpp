#include <cstdio>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.1415926


using namespace std;
using namespace cv;


class LineDetection {

public:
    Mat getLinesFromImage(Mat &image) {
        return getLinesFromImage(image, false);
    }

    Mat getLinesFromImage(Mat &image, bool showSteps) {

        // parameters
        int cannyLowerThreshold = 80;
        int cannyUpperThreshold = cannyLowerThreshold * 2;
        int threshThreshold = 170;
        int threshMaxval = 255;
        int houghPMinLineLength = 10;
        int houghPGap = 20;
        int houghPMinVote = 30;
        double houghPDeltaRho = 1;
        double houghPDeltaTheta = PI/180;

        // set the ROI for the image
        Rect roi(0, image.cols / 4, image.cols - 1, image.rows - image.cols / 4);
        Mat imgROI = image(roi);

        // Canny algorithm
        Mat contours;
        Canny(imgROI, contours, cannyLowerThreshold, cannyUpperThreshold, 3);
        Mat contoursInv;
        threshold(contours, contoursInv, threshThreshold, threshMaxval, THRESH_BINARY_INV);

        // Do probabilistic Hough
        Mat houghP(imgROI.size(), CV_8U, Scalar(0));
        vector<Vec4i> lines;

        HoughLinesP(contours,lines,houghPDeltaRho,houghPDeltaTheta,houghPMinVote, houghPMinLineLength, houghPGap);
        drawDetectedLines(houghP, lines);


        if(showSteps) {
            namedWindow( "Input", CV_WINDOW_AUTOSIZE );
            imshow( "Input", image );
            namedWindow( "Canny", CV_WINDOW_AUTOSIZE );
            imshow( "Canny", contours );
            namedWindow( "threshold", CV_WINDOW_AUTOSIZE );
            imshow( "threshold", contoursInv );
            namedWindow( "HoughLinesP", CV_WINDOW_AUTOSIZE );
            imshow( "HoughLinesP", houghP );
        }

        return houghP;
    };

    // Draw the detected lines on an image
    void drawDetectedLines(Mat &image, vector<Vec4i> & lines, cv::Scalar color=cv::Scalar(255)) {

        // Draw the lines
        vector<Vec4i>::const_iterator it2= lines.begin();

        while (it2!=lines.end()) {

            Point pt1((*it2)[0],(*it2)[1]);
            Point pt2((*it2)[2],(*it2)[3]);

            line( image, pt1, pt2, color, 1 );
            ++it2;
        }
    }

};