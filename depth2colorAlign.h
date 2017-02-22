#ifndef DEPTH_2_COLOR_ALIGN
#define DEPTH_2_COLOR_ALIGN

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

void depth2colorAlign(const cv::Mat &im_color, const cv::Mat &im_depth, const cv::Mat Transform, 
	const cv::Mat K_color, const cv::Mat K_depth, cv::Mat &im_registrated_depth);

void registration(const cv::Mat &im_color, const cv::Mat &pointclouds, 
	const cv::Mat &Transform, const cv::Mat &K, cv::Mat &im_registrated_depth);

void prepare_depth_uvs(const int width, const int height, cv::Mat &uvs);

void depthIm2pointclouds(const cv::Mat &im_depth, const cv::Mat &K, cv::Mat &pointclouds);

#endif