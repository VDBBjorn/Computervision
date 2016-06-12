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


        // Filter hough detected lines by angle
        int minAngle = 20;
        int maxAngle = 80;

        // parameters
        int cannyLowerThreshold = 40; // 40
        int cannyUpperThreshold = cannyLowerThreshold * 2;
        int houghPMinLineLength = 7;
        int houghPGap = 10;
        int houghPMinVote = 7;
        double houghPDeltaRho = 1;
        double houghPDeltaTheta = PI / 180;

        // set the ROI for the image
        Rect roi(0, image.cols / 4, image.cols - 1, image.rows - image.cols / 4);
        Mat imgROI = image(roi);


        Mat tmp;
        Size size;
        size.width = 100; // 100
        size.height = 30; // 20
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, size);
        erode(imgROI, tmp, kernel);

        size.width = 85; // 85
        size.height = 25; // 18
        kernel = getStructuringElement(MORPH_ELLIPSE, size);
        dilate(tmp, tmp, kernel);


        // Canny algorithm
        Mat contours;
        Mat result(imgROI.size(), CV_8U, Scalar(0));
        Canny(tmp, contours, cannyLowerThreshold, cannyUpperThreshold, 3);
        Mat contoursInv;
        contours.copyTo(result);

        // Do probabilistic Hough
        vector<Vec4i> lines;

        HoughLinesP(contours,lines,houghPDeltaRho,houghPDeltaTheta,houghPMinVote, houghPMinLineLength, houghPGap);
        //drawDetectedLines(result, lines);


        if (showSteps) {
            namedWindow("Input", CV_WINDOW_AUTOSIZE);
            imshow("Input", tmp);
            namedWindow("Canny", CV_WINDOW_AUTOSIZE);
            imshow("Canny", contours);
            /*  namedWindow( "threshold", CV_WINDOW_AUTOSIZE );
              imshow( "threshold", contoursInv ); */
            /* namedWindow( "HoughLinesP", CV_WINDOW_AUTOSIZE );
             imshow( "HoughLinesP", houghP );*/
            namedWindow("result", CV_WINDOW_AUTOSIZE);
            imshow("result", result);
        }

        return result;
    };

    // Draw the detected lines on an image
    void drawDetectedLines(Mat &image, vector<Vec4i> &lines, cv::Scalar color = cv::Scalar(255)) {

        // Draw the lines
        vector<Vec4i>::const_iterator it2 = lines.begin();

        while (it2 != lines.end()) {

            Point pt1((*it2)[0], (*it2)[1]);
            Point pt2((*it2)[2], (*it2)[3]);

            line(image, pt1, pt2, color, 1);
            ++it2;
        }
    }

};