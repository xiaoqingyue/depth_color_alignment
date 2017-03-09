#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "depth2colorAlign.h"
#include <ctime>

int main(int argc, char* argv[])
{
	cv::Mat im_color = cv::imread("../images/rgb_3.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat im_depth = cv::imread("../images/depth_3.png", CV_LOAD_IMAGE_ANYDEPTH);

	im_color.convertTo(im_color, CV_16UC3);
	im_color = im_color * 255;

	// cv::namedWindow("color", CV_WINDOW_NORMAL);
	// imshow("color", im_color);
	// cv::namedWindow("depth", CV_WINDOW_NORMAL);
	// imshow("depth", 4 * im_depth);
	// cv::waitKey();

	cv::Mat im_registrated_depth;

	cv::Mat Transform = (cv::Mat_<double>(4, 4)
		<< 1, 0, 0, 0.00683648,
		   0, 1, 0, -0.000771234,
		   0, 0, 1, -0.000460127,
		   0, 0, 0, 1);

	cv::Mat K_color = (cv::Mat_<double>(3, 3)
		<< 741.029, 0, 307.898,
		   0, 741.029, 236.921,
		   0, 0, 1);

	cv::Mat K_depth = (cv::Mat_<double>(3, 3)
		<< 584.615, 0, 303.407,
		   0, 584.615, 227.556,
		   0, 0, 1);

	const clock_t begin_time = clock();

	cv::Size color_size(im_color.cols, im_color.rows);
	cv::Size depth_size(im_depth.cols, im_depth.rows);

	Depth2ColorAlign app(color_size, depth_size, Transform, K_color, K_depth);

	app.align(im_depth, im_registrated_depth);

	std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000.0 << "ms" << std::endl;
	// std::cout << im_registrated_depth << std::endl;

	cv::namedWindow("registrated depth", CV_WINDOW_NORMAL);
	imshow("registrated depth", im_registrated_depth * 8);
	cvtColor(im_registrated_depth, im_registrated_depth, CV_GRAY2BGR);

	cv::Mat colorAddDepth = im_color / 4  + im_registrated_depth * 8;
	cv::namedWindow("add", CV_WINDOW_NORMAL);
	imshow("add", colorAddDepth);
	imwrite("../images/registrated.jpg", colorAddDepth);

	cv::waitKey();

	return 0;
}