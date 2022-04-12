#include "pch.h"
#include "sharpness.h"

using namespace std;
using namespace cv;

// 删除清晰度函数体内的灰度化

/**
* Brenner梯度方法
*
* Inputs:
* @param image:
* Return: double
*/
double brenner(cv::Mat& gray_img)
{
	//assert(image.empty());
	double result = .0f;
	for (int i = 0; i < gray_img.rows; ++i) {
		uchar* data = gray_img.ptr<uchar>(i);
		for (int j = 0; j < gray_img.cols - 2; ++j) {
			result += pow(data[j + 2] - data[j], 2);
		}
	}
	return result / gray_img.total();
}


/**
* Tenengrad梯度方法
*
* Inputs:
* @param image:
* Return: double
*/
double tenengard(cv::Mat& gray_img)
{
	//assert(image.empty());
	
	cv::Mat res_img, sobel_x, sobel_y, G;
	
	//medianBlur(gray_img, res_img, 3);

	//分别计算x/y方向梯度
	//cv::Sobel(res_img, sobel_x, CV_32FC1, 1, 0);
	//cv::Sobel(res_img, sobel_y, CV_32FC1, 0, 1);

	cv::Sobel(gray_img, sobel_x, CV_32FC1, 1, 0);
	cv::Sobel(gray_img, sobel_y, CV_32FC1, 0, 1);
	
	cv::multiply(sobel_x, sobel_x, sobel_x);
	cv::multiply(sobel_y, sobel_y, sobel_y);
	cv::Mat sqrt_mat = sobel_x + sobel_y;
	cv::sqrt(sqrt_mat, G);

	return cv::mean(G)[0];
}

/**
* Laplacian 梯度函数
*
* Inputs:
* @param image:
* Return: double
*/
double laplacian(cv::Mat& gray_img)
{
	//assert(image.empty());

	cv::Mat lap_image;
	cv::Laplacian(gray_img, lap_image, CV_32FC1);
	lap_image = cv::abs(lap_image);

	return cv::mean(lap_image)[0];
}

/**
* SMD（灰度方差）函数
*
* Inputs:
* @param image:
* Return: double
*/
double smd(cv::Mat& gray_img)
{
	//assert(image.empty());

	cv::Mat res_img, smd_image_x, smd_image_y, G;
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(0, 1) = -1.0;
	kernel_y.at<float>(1, 1) = 1.0;
	cv::filter2D(gray_img, smd_image_x, gray_img.depth(), kernel_x);
	cv::filter2D(gray_img, smd_image_y, gray_img.depth(), kernel_y);

	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	G = smd_image_x + smd_image_y;

	return cv::mean(G)[0];
}


/**
* SMD2 （灰度方差乘积）函数
*
* Inputs:
* @param image:
* Return: double
*/
double smd2(cv::Mat& gray_img)
{
	//assert(image.empty());

	cv::Mat smd_image_x, smd_image_y, G;
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	cv::filter2D(gray_img, smd_image_x, gray_img.depth(), kernel_x);
	cv::filter2D(gray_img, smd_image_y, gray_img.depth(), kernel_y);

	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	cv::multiply(smd_image_x, smd_image_y, G);

	return cv::mean(G)[0];
}

/**
* 能量梯度函数
*
* Inputs:
* @param image:
* Return: double
*/
double energy_gradient(cv::Mat& gray_img)
{
	//assert(image.empty());

	cv::Mat smd_image_x, smd_image_y, G;
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	cv::filter2D(gray_img, smd_image_x, gray_img.depth(), kernel_x);
	cv::filter2D(gray_img, smd_image_y, gray_img.depth(), kernel_y);

	cv::multiply(smd_image_x, smd_image_x, smd_image_x);
	cv::multiply(smd_image_y, smd_image_y, smd_image_y);
	G = smd_image_x + smd_image_y;

	return cv::mean(G)[0];
}

/**
* EAV点锐度算法函数
*
* Inputs:
* @param image:
* Return: double
*/
double eav(cv::Mat& gray_img)
{
	//assert(image.empty());

	cv::Mat smd_image_x, smd_image_y, G;
	
	double result = .0f;
	for (int i = 1; i < gray_img.rows - 1; ++i) {
		uchar* prev = gray_img.ptr<uchar>(i - 1);
		uchar* cur = gray_img.ptr<uchar>(i);
		uchar* next = gray_img.ptr<uchar>(i + 1);
		for (int j = 0; j < gray_img.cols; ++j) {
			result += (abs(prev[j - 1] - cur[i]) * 0.7 + abs(prev[j] - cur[j]) + abs(prev[j + 1] - cur[j]) * 0.7 +
				abs(next[j - 1] - cur[j]) * 0.7 + abs(next[j] - cur[j]) + abs(next[j + 1] - cur[j]) * 0.7 +
				abs(cur[j - 1] - cur[j]) + abs(cur[j + 1] - cur[j]));
		}
	}
	return result / gray_img.total();
}

/**
* FC 灰度方差法
*
* Inputs:
* @param image:
* Return: double
*/
double FC(cv::Mat& gray_img)
{
	Mat mat_mean, mat_stddev;
	
	meanStdDev(gray_img, mat_mean, mat_stddev); //求灰度图像的均值、均方差
	double m = mat_mean.at<double>(0, 0);
	double s = mat_stddev.at<double>(0, 0);
	return s;
}

double cuda_FC(cv::Mat& image)
{
	cv::Scalar mean, std;
	cv::cuda::GpuMat src, gray;
	src.upload(image);
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	cv::cuda::meanStdDev(gray, mean, std);
	return std.val[0];
}

