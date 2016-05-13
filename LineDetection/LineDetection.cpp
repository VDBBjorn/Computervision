#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include "LineFinder.h"

using namespace std;
using namespace cv;


class LineDetection {

public:
    Mat getLinesFromImage(Mat &image, int initialHoughVote, int &houghVote, int initialHoughVote2, int &houghVote2,
                          bool drawLinesOnImage, bool showSteps, bool &isReliable) {

        // Filter hough detected lines by angle
        int minAngle = 20;
        int maxAngle = 80;

        // set the ROI for the image
        Rect roi(0, image.cols / 4, image.cols - 1, image.rows - image.cols / 4);
        Mat imgROI = image(roi);
        Mat secondHough(imgROI.size(), CV_8U, Scalar(0));

        // Canny algorithm
        Mat contours;
        Canny(imgROI, contours, 50, 250);
        Mat contoursInv;
        threshold(contours, contoursInv, 128, 255, THRESH_BINARY_INV);

        // Display Canny image
      /*  if (showSteps) {
            namedWindow("Contours");
            imshow("Contours1", contoursInv);
        }*/

        std::vector<Vec4i> li;

        // Create LineFinder instance
        LineFinder ld;

        // Set probabilistic Hough parameters
        ld.setLineLengthAndGap(60, 30);
        ld.setMinVote(2);

        // Detect lines
        li = ld.findLines(contours);

        Mat houghP(imgROI.size(), CV_8U, Scalar(0));
        ld.setShift(0);
        ld.drawDetectedLines(houghP);

        if (showSteps) {
            namedWindow("Detected Lines with HoughP");
            imshow("Detected Lines with HoughP", houghP);
            //imwrite("houghP.bmp", houghP);
        }

        /*
           Hough tranform for line detection with feedback
           Increase by 25 for the next frame if we found some lines.
           This is so we don't miss other lines that may crop up in the next frame
           but at the same time we don't want to start the feed back loop from scratch.
       */
        std::vector<Vec2f> lines;

        Mat result(imgROI.size(), CV_8U, Scalar(255));
        imgROI.copyTo(result);

        // Draw the limes
        std::vector<Vec2f>::const_iterator it = lines.begin();
        Mat hough(imgROI.size(), CV_8U, Scalar(0));

        HoughLines(contours,lines,1,PI/180, houghVote);

        vector<RoadLine> resultLines(2);

        float largestXLeft = 0;
        float smallestXRight = result.cols;

        while (it != lines.end()) {

            float rho = (*it)[0];   // first element is distance rho
            float theta = (*it)[1]; // second element is angle theta

            float xStart = (rho - result.rows * sin(theta)) / cos(theta);

            //point of intersection of the line with first row
            Point pt1(rho / cos(theta), 0);

            // point of intersection of the line with last row
            Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);

            double ori1 =
                    (atan2(static_cast<double>(pt1.y - pt2.y), static_cast<double>(pt1.x - pt2.y)) + PI / 2) * 180 / PI;

            float x = (rho - result.rows * sin(theta)) / cos(theta);

            if (ori1 < maxAngle && ori1 > minAngle && x >= 0 && x <=
                                                                result.cols) { // filter to remove vertical and horizontal lines // || theta < 3.14 && theta > 1.66


                line(hough, pt1, pt2, Scalar(0), 8);
                line(result, pt1, pt2, Scalar(0), 8);

                pt1.y += image.cols / 4;
                pt2.y += image.cols / 4;

                //line( image, pt1, pt2, Scalar(0), 8);
                if (x < result.cols / 2 && x > largestXLeft && ori1 < 45) {
                    largestXLeft = x;
                    RoadLine l(pt1, pt2);
                    resultLines[0] = l;
                } else if (x > result.cols / 2 && x < smallestXRight && ori1 < 45) {
                    smallestXRight = x;
                    RoadLine l(pt1, pt2);
                    resultLines[1] = l;
                }

            }

            ++it;
        }

        Mat houghPinv(imgROI.size(), CV_8U, Scalar(0));

        if (resultLines[0].pt1.x != 0 && resultLines[0].pt1.y != 0
            && resultLines[0].pt2.x != 0 && resultLines[0].pt2.y != 0
            && resultLines[1].pt1.x != 0 && resultLines[1].pt1.y != 0
            && resultLines[1].pt2.x != 0 && resultLines[1].pt2.y != 0) {
            isReliable = true;
            cout << "Reliable! :-)" << endl;
            resultLines[0].pt1.y -= image.cols / 4;
            resultLines[0].pt2.y -= image.cols / 4;
            resultLines[1].pt1.y -= image.cols / 4;
            resultLines[1].pt2.y -= image.cols / 4;
            line(hough, resultLines[0].pt1, resultLines[0].pt2, Scalar(255), 8);
            line(hough, resultLines[1].pt1, resultLines[1].pt2, Scalar(255), 8);
        }

        // Display the detected line image
        if (showSteps) {
            namedWindow("Detected Lines with Hough");
            imshow("Detected Lines with Hough", hough);
        }

        // bitwise OR of the two hough images
        bitwise_or(houghP, hough, houghP);

        Mat dst(imgROI.size(), CV_8U, Scalar(0));
        threshold(houghP, houghPinv, 150, 255, THRESH_BINARY_INV); // threshold and invert to black lines

        if (showSteps) {
            namedWindow("Detected Lines with Bitwise");
            imshow("Detected Lines with Bitwise", houghPinv);
            waitKey(0);
        }

        isReliable = false;
        //return resultLines;
        return houghPinv;
    };

};