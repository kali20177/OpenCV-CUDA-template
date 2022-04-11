#include "pch.h"

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;

void testcuda_rgb2gray()
{
	cv::cuda::GpuMat src, gray;
	for (int i = 0; i < 150; ++i)
	{
		string path = "E:/MyData/2022-01-21/卵巢囊性纤维瘤/" + to_string(i) + ".bmp";

		Mat src_host = imread(path, 1);
		src.upload(src_host);
		cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
		Mat gray_host;
		gray.download(gray_host);
	}
}

void test_rgb2gray()
{
	Mat image_gray, imageres;
	for (int i = 0; i < 150; ++i)
	{
		string path = "E:/MyData/2022-01-21/卵巢囊性纤维瘤/" + to_string(i) + ".bmp";

		Mat image = imread(path, 1);
		cvtColor(image, image_gray, CV_BGR2GRAY);
	}
}

int main()
{
	auto start = std::chrono::steady_clock::now();
	testcuda_rgb2gray();
	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	
	cout << "cuda time consume: " << duration.count() / 150 << "ms" << endl;


	auto start1 = std::chrono::steady_clock::now();
	test_rgb2gray();
	auto end1 = std::chrono::steady_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

	cout << "cpu time consume: " << duration1.count() / 150 << "ms" << endl;
	
	// 显卡信息
	/*cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = cuda::getCudaEnabledDeviceCount();
	printf("GPU Device Count : %d\n", count);
	printf("OpenCV version: %s\n", CV_VERSION);*/
}