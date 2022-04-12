#pragma once

#include "pch.h"

double brenner(cv::Mat& image);
double tenengard(cv::Mat& image);
double laplacian(cv::Mat& image);
double smd(cv::Mat& image);
double smd2(cv::Mat& image);
double energy_gradient(cv::Mat& image);
double eav(cv::Mat& image);
double FC(cv::Mat& image);
double cuda_FC(cv::Mat& image);
