#include "depth2colorAlign.h"

void depth2colorAlign(const cv::Mat &im_color, const cv::Mat &im_depth, cv::Mat &im_registrated_depth)
{
	cv::Mat Transform = (cv::Mat_<double>(4, 4)
		<< 1, 0, 0, -0.00683648,
		   0, 1, 0, 0.000771234,
		   0, 0, 1, 0.000460127,
		   0, 0, 0, 1);

	cv::Mat K_color = (cv::Mat_<double>(3, 3)
		<< 741.029, 0, 307.898,
		   0, 741.029, 236.921,
		   0, 0, 1);

	cv::Mat K_depth = (cv::Mat_<double>(3, 3)
		<< 584.615, 0, 303.407,
		   0, 584.615, 227.556,
		   0, 0, 1);

	cv::Mat pointclouds;

	depthIm2pointclouds(im_depth, K_depth, pointclouds);

	// std::cout << pointclouds << std::endl;

	registration(im_color, pointclouds, Transform, K_color, im_registrated_depth);
}

/**
/* @brief covert depth image to pointclouds
/* @param K - the intrinc parameter K of depth camera
/* @param pointclouds - 3 x N matrix, each column is on point
/* @version Shon Xiao, 2017/2/17
*/
void depthIm2pointclouds(const cv::Mat &im_depth, const cv::Mat &K, cv::Mat &pointclouds)
{
	int width = im_depth.cols;
	int height = im_depth.rows;
	cv::Mat im_depth_m;
	im_depth.convertTo(im_depth_m, CV_64FC1, 0.001); // change the unit from mm to m
	cv::Mat uvs;
	prepare_depth_uvs(width, height, uvs);
	uvs.convertTo(uvs, CV_64FC1);
	// std::cout << uvs << std::endl;
	cv::Mat Zc = im_depth_m.reshape(1, 1);
	// std::cout << Zc << std::endl;
	cv::Mat tmp = K.inv() * uvs;
	pointclouds = cv::Mat::zeros(3, tmp.cols, tmp.type());
	// std::cout << Zc.rows << " " << Zc.cols << std::endl;
	//Zc.convertTo(Zc, CV_64FC1);
	for(int i = 0; i < 3; ++i)
	{
		cv::Mat tmp_row = tmp.row(i).mul(Zc);
		tmp_row.copyTo(pointclouds.row(i));
	}
}

/**
/* @brief registriate and get depth of each color pixels
/* @param Transform - transform the 3d pointclouds to color image coordinate
/* @param K - the intrinc parameter of color camera
/* @param pointclouds - 3 x N matrix, each column as one point
/* @version Shon Xiao, 2017/2/16
*/
void registration(const cv::Mat &im_color, const cv::Mat &pointclouds, 
	const cv::Mat &Transform, const cv::Mat &K, cv::Mat &im_registrated_depth)
{
	cv::Mat points3d = pointclouds.clone();
	cv::Mat onesM = cv::Mat::ones(1, points3d.cols, points3d.type());
	points3d.push_back(onesM);
	cv::Mat Pc = Transform(cv::Rect(0, 0, 4, 3)) * points3d;
	// std::cout << Pc << std::endl;
	cv::Mat uv_h = K * Pc;
	// std::cout << uv_h << std::endl;
	cv::Mat uv(2, uv_h.cols, uv_h.type());
	// convert homogeneous 
	cv::Mat tmp;
	divide(uv_h.row(0), uv_h.row(2), tmp);
	tmp.copyTo(uv.row(0));
	divide(uv_h.row(1), uv_h.row(2), tmp);
	tmp.copyTo(uv.row(1));
	// std::cout << uv.type() << std::endl << uv << std::endl;
	// std::cout << Pc.type() << " " << Pc.rows << " " << Pc.cols << std::endl;
	// std::cout << Pc.row(3) << std::endl;
	im_registrated_depth = cv::Mat::zeros(im_color.rows, im_color.cols, CV_16UC1);
	for(int i = 0; i < uv.cols; ++i)
	{
		int u = uv.at<double>(0, i);
		int v = uv.at<double>(1, i);
		if(v >= 0 && v < im_color.rows && u >= 0 && u < im_color.cols)
		{
			// std::cout << u << " " << v << std::endl;
			int depth = Pc.at<double>(2, i) * 1000.0; // unit mm
			// std::cout << i << " " << Pc.at<double>(3, i) << std::endl;
			if(depth > 65535)
			{
				depth = 65535;
			}
			else if(depth < 0)
			{
				depth = 0;
			}
			im_registrated_depth.at<unsigned short>(v, u) = depth;
		}
	}
}


void meshgrid(const int width, const int height, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> vx, vy;
	for(int i = 0; i < width; ++i)
	{
		vx.push_back(i);
	}
	for(int i = 0; i < height; ++i)
	{
		vy.push_back(i);
	}

	repeat(cv::Mat(vx).t(), height, 1, X);
	repeat(cv::Mat(vy), 1, width, Y);
}

void prepare_depth_uvs(const int width, const int height, cv::Mat &uvs)
{
	cv::Mat X, Y;
	meshgrid(width, height, X, Y);
	cv::Mat tmpX, tmpY;
	tmpX = X.reshape(1, 1);
	tmpY = Y.reshape(1, 1);
	uvs = tmpX;
	// std::cout << "tmpX " << tmpX.rows << " " << tmpX.cols << std::endl;
	// std::cout << "tmpY " << tmpY.rows << " " << tmpY.cols << std::endl;
	// std::cout << uvs.cols << " " << uvs.type() << std::endl;
	uvs.push_back(tmpY);
	cv::Mat onesM = cv::Mat::ones(1, uvs.cols, uvs.type());
	uvs.push_back(onesM);
}