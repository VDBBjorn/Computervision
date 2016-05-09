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
    Mat getLinesFromImage(Mat & image,int initialHoughVote, int & houghVote,int initialHoughVote2, int & houghVote2, bool drawLinesOnImage, bool showSteps, bool & isReliable) {

			// Filter hough detected lines by angle
			int minAngle = 20;
			int maxAngle = 80;

			// set the ROI for the image
			Rect roi(0,image.cols/4,image.cols-1,image.rows - image.cols/4);
			Mat imgROI = image(roi);
			Mat secondHough(imgROI.size(),CV_8U,Scalar(0));

			// Canny algorithm
			Mat contours;
			Canny(imgROI,contours,50,250);
			Mat contoursInv;
			threshold(contours,contoursInv,128,255,THRESH_BINARY_INV);

		   // Display Canny image
			if(showSteps){
				namedWindow("Contours");
				imshow("Contours1",contoursInv);
			}

		 /* 
			Hough tranform for line detection with feedback
			Increase by 25 for the next frame if we found some lines.  
			This is so we don't miss other lines that may crop up in the next frame
			but at the same time we don't want to start the feed back loop from scratch. 
		*/
			std::vector<Vec2f> lines;
		/*	if (houghVote < 1 or lines.size() > 2){ // we lost all lines. reset
				houghVote = initialHoughVote; 
			}
			else{ houghVote += 25;} */
			//while(lines.size() < 20 && houghVote > 0){
				HoughLines(contours,lines,1,PI/180, houghVote);
			//	houghVote -= 5;
			//}
			Mat result(imgROI.size(),CV_8U,Scalar(255));
			imgROI.copyTo(result);

		   // Draw the limes
			std::vector<Vec2f>::const_iterator it= lines.begin();
			Mat hough(imgROI.size(),CV_8U,Scalar(0));
			while (it!=lines.end()) {

				float rho= (*it)[0];   // first element is distance rho
				float theta= (*it)[1]; // second element is angle theta

                float xStart = (rho-result.rows*sin(theta))/cos(theta);

				//point of intersection of the line with first row
				Point pt1(rho/cos(theta),0);

				// point of intersection of the line with last row
				Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);

                double ori1= (atan2(static_cast<double>(pt1.y-pt2.y),static_cast<double>(pt1.x-pt2.y))+PI/2) * 180/PI;

               /* printf("angle= %f\n", ori1);
				line( result, pt1, pt2, Scalar(255), 8);
				namedWindow("Detected Lines with Hough");
				imshow("Detected Lines with Hough",result);
				waitKey(0);
				destroyAllWindows();*/

				if ( ori1 < maxAngle && ori1 > minAngle && xStart >= 0 && xStart <= result.cols) { // filter to remove lines with wrong angle

					// draw a white line
					line( result, pt1, pt2, Scalar(255), 8); 
					line( hough, pt1, pt2, Scalar(255), 8);
				}
				
				++it;
			}

		    // Display the detected line image
			if(showSteps){
				namedWindow("Detected Lines with Hough");
				imshow("Detected Lines with Hough",result);
			}
			
			std::vector<Vec4i> li;

		   // Create LineFinder instance
			LineFinder ld;

		   // Set probabilistic Hough parameters
			ld.setLineLengthAndGap(100,30);
			ld.setMinVote(2);
			
			// Detect lines
			li= ld.findLines(contours);	
		   
			Mat houghP(imgROI.size(),CV_8U,Scalar(0));
			ld.setShift(0);
			ld.drawDetectedLines(houghP);

			if(showSteps){
				namedWindow("Detected Lines with HoughP");
				imshow("Detected Lines with HoughP", houghP);
				//imwrite("houghP.bmp", houghP);
			}

		   // bitwise AND of the two hough images
			bitwise_or(houghP,hough,houghP);
			Mat houghPinv(imgROI.size(),CV_8U,Scalar(0));
			Mat dst(imgROI.size(),CV_8U,Scalar(0));
			threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines

			if(showSteps){
				namedWindow("Detected Lines with Bitwise");
				imshow("Detected Lines with Bitwise", houghPinv);
			}

			Canny(houghPinv,contours,100,350);
			li= ld.findLines(contours);

            // Display Canny image
			if(showSteps){
				namedWindow("Contours");
				imshow("Contours2",contours);
				//imwrite("contours.bmp", contoursInv);
			}




			vector<vector<Point> > contourPoints;
			vector<Vec4i> hierarchy;
			findContours( contours, contourPoints, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		   // Set probabilistic Hough parameters
			/*ld.setLineLengthAndGap(150,10);
			ld.setMinVote(3);
		//	ld.setShift(image.cols/3);
			ld.drawDetectedLines(secondHough, Scalar(255));

			if(showSteps){
				namedWindow("Result");
				imshow("Result", secondHough);
				//waitKey(0);
				//destroyAllWindows();
				//imwrite("houghP.bmp", houghP);
			}*/

			Mat result2(imgROI.size(),CV_8U,Scalar(255));

			lines.clear();
			houghVote2 = initialHoughVote2;
		/*	if (houghVote2 < 1 or lines.size() > 2){ // we lost all lines. reset
				houghVote2 = initialHoughVote2; 
			}
			else{ houghVote2 += 25;
			} */
			//while(lines.size() < 20 && houghVote2 > 0){
				HoughLines(contours,lines,1,PI/180, houghVote2);
				//houghVote2 -= 5;
			//}

			isReliable = false;
			imgROI.copyTo(result2);
            vector<RoadLine> resultLines(2);
			if(lines.size() < 2) {
				cout << "Bocht";
				ld.setLineLengthAndGap(140,30);
				ld.setMinVote(3);
				ld.setShift(image.cols/4);
				ld.drawDetectedLines(result2, Scalar(255));
			} else {
			   // Draw the limes
				it= lines.begin();
				Mat hough2(imgROI.size(),CV_8U,Scalar(0));

                float largestXLeft = 0;
                float smallestXRight = result.cols;

				while (it!=lines.end()) {

					float rho= (*it)[0];   // first element is distance rho
					float theta= (*it)[1]; // second element is angle theta

					//point of intersection of the line with first row
					Point pt1(rho/cos(theta),0);        
					// point of intersection of the line with last row
					Point pt2((rho-result2.rows*sin(theta))/cos(theta),result2.rows);
					
					double ori1= (atan2(static_cast<double>(pt1.y-pt2.y),static_cast<double>(pt1.x-pt2.y))+PI/2) * 180/PI;
                    float x = (rho-result.rows*sin(theta))/cos(theta);

					if ( ori1 < maxAngle && ori1 > minAngle && x >= 0 && x <= result.cols) { // filter to remove vertical and horizontal lines // || theta < 3.14 && theta > 1.66


						line( hough2, pt1, pt2, Scalar(0), 8);
						line( result2, pt1, pt2, Scalar(0), 8);

                        pt1.y += image.cols/4;
                        pt2.y += image.cols/4;

                        //line( image, pt1, pt2, Scalar(0), 8);
                        if(x < result.cols / 2 && x > largestXLeft && ori1 < 45) {
                            largestXLeft = x;
                            RoadLine l(pt1, pt2);
                            resultLines[0] = l;
                        } else if(x > result.cols / 2 && x < smallestXRight && ori1 < 45) {
                            smallestXRight = x;
                            RoadLine l(pt1, pt2);
                            resultLines[1] = l;
                        }

					}

                    ++it;
				}

                line( image, resultLines[0].pt1, resultLines[0].pt2, Scalar(0), 8);
                line( image, resultLines[1].pt1, resultLines[1].pt2, Scalar(0), 8);

                if(resultLines[0].pt1.x != 0 && resultLines[0].pt1.y != 0 
                	&& resultLines[0].pt2.x != 0 && resultLines[0].pt2.y != 0
                	&& resultLines[1].pt1.x != 0 && resultLines[1].pt1.y != 0
                	&& resultLines[1].pt2.x != 0 && resultLines[1].pt2.y != 0){
		                isReliable = true;
		            	resultLines[0].pt1.y -= image.cols/4;
		            	resultLines[0].pt2.y -= image.cols/4;
		            	resultLines[1].pt1.y -= image.cols/4;
		            	resultLines[1].pt2.y -= image.cols/4;
		                line( houghPinv, resultLines[0].pt1, resultLines[0].pt2, Scalar(0), 8);
		                line( houghPinv, resultLines[1].pt1, resultLines[1].pt2, Scalar(0), 8);
            	}
			}

			
			if(showSteps){
				namedWindow("Detected Lines with Hough2");
				imshow("Detected Lines with Hough2",result2);
				waitKey(0);
				destroyAllWindows();
				//imwrite("hough.bmp", result);
			}


    		//return resultLines;
			return houghPinv;
		};

};