#ifndef DEPTH_2_COLOR_ALIGN
#define DEPTH_2_COLOR_ALIGN

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <numeric>

class Depth2ColorAlign
{
public:
	Depth2ColorAlign(const cv::Size &colorImageSize, const cv::Size &depthImageSize, const cv::Mat &Transform, 
		const cv::Mat &K_color, const cv::Mat &K_depth);

	~Depth2ColorAlign() {};

	void align(const cv::Mat &im_depth, cv::Mat &im_registrated_depth);

private:
	cv::Mat inv_Kd_x_uvs;
	cv::Mat Kc_x_RT;
	cv::Size colorSize;

	void mapDepth(const cv::Mat &uv, cv::Mat &im_registrated_depth);

	void prepare_depth_uvs(const int width, const int height, cv::Mat &uvs);

	void meshgrid(const int width, const int height, cv::Mat &X, cv::Mat &Y);

};

#endif