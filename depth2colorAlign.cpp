#include "depth2colorAlign.h"


Depth2ColorAlign::Depth2ColorAlign(const cv::Size &colorImageSize, const cv::Size &depthImageSize, const cv::Mat &Transform, 
		const cv::Mat &K_color, const cv::Mat &K_depth): colorSize(colorImageSize)
{
	cv::Mat uvs;
	prepare_depth_uvs(depthImageSize.width, depthImageSize.height, uvs);
	uvs.convertTo(uvs, CV_64FC1);
	inv_Kd_x_uvs = K_depth.inv() * uvs;
	Kc_x_RT = K_color * Transform(cv::Rect(0, 0, 4, 3));
}

void Depth2ColorAlign::align(const cv::Mat &im_depth, cv::Mat &im_registrated_depth)
{
	cv::Mat im_depth_m;
	im_depth.convertTo(im_depth_m, CV_64FC1, 0.001); // change the unit from mm to m
	cv::Mat Zc = im_depth_m.reshape(1, 1);

	cv::Mat pointclouds = cv::Mat::ones(4, inv_Kd_x_uvs.cols, inv_Kd_x_uvs.type());
	// std::cout << Zc.rows << " " << Zc.cols << std::endl;
	//Zc.convertTo(Zc, CV_64FC1);
	for(int i = 0; i < 3; ++i)
	{
		cv::Mat tmp_row = inv_Kd_x_uvs.row(i).mul(Zc);
		tmp_row.copyTo(pointclouds.row(i));
	}
	// std::cout << Pc << std::endl;
	cv::Mat uv_h = Kc_x_RT * pointclouds;
	// std::cout << uv_h << std::endl;
	// convert homogeneous 
	cv::Mat tmp;
	divide(uv_h.row(0), uv_h.row(2), tmp);
	tmp.copyTo(uv_h.row(0));
	divide(uv_h.row(1), uv_h.row(2), tmp);
	tmp.copyTo(uv_h.row(1));

	mapDepth(uv_h, im_registrated_depth);
}


void Depth2ColorAlign::mapDepth(const cv::Mat &uv, cv::Mat &im_registrated_depth)
{
	im_registrated_depth = cv::Mat::zeros(colorSize.height, colorSize.width, CV_16UC1);
	// map the depth
	const double *pu = uv.ptr<double> (0);
	const double *pv = uv.ptr<double> (1);
	const double *pdepth = uv.ptr<double> (2);
	for(int i = 0; i < uv.cols; ++i)
	{

		// int u = uv.at<double>(0, i);
		int u = *pu++;
		int v = *pv++;

		if(v >= 0 && v < colorSize.height && u >= 0 && u < colorSize.width)
		{
			// std::cout << u << " " << v << std::endl;
			int depth = *pdepth * 1000.0; // unit mm
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
		pdepth++;
	}
}


void Depth2ColorAlign::meshgrid(const int width, const int height, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> vx(width), vy(height);
	std::iota(vx.begin(), vx.end(), 0);
	std::iota(vy.begin(), vy.end(), 0);

	repeat(cv::Mat(vx).t(), height, 1, X);
	repeat(cv::Mat(vy), 1, width, Y);
}

void Depth2ColorAlign::prepare_depth_uvs(const int width, const int height, cv::Mat &uvs)
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