#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "depth2colorAlign.h"

int main(int argc, char* argv[])
{
	cv::Mat im_color = cv::imread("../images/rgb_3.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat im_depth = cv::imread("../images/depth_3.png", CV_LOAD_IMAGE_ANYDEPTH);

	im_color.convertTo(im_color, CV_16UC1);
	im_color = im_color * 255;

	cv::namedWindow("color", CV_WINDOW_NORMAL);
	imshow("color", im_color);
	cv::namedWindow("depth", CV_WINDOW_NORMAL);
	imshow("depth", 4 * im_depth);
	cv::waitKey();

	cv::Mat im_registrated_depth;

	depth2colorAlign(im_color, im_depth, im_registrated_depth);

	// std::cout << im_registrated_depth << std::endl;

	cv::namedWindow("registrated depth", CV_WINDOW_NORMAL);
	imshow("registrated depth", im_registrated_depth * 16);
	cvtColor(im_registrated_depth, im_registrated_depth, CV_GRAY2BGR);

	cv::Mat colorAddDepth = im_color / 4  + im_registrated_depth * 8;
	cv::namedWindow("add", CV_WINDOW_NORMAL);
	imshow("add", colorAddDepth);

	cv::waitKey();

	return 0;
}