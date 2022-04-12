#include "pch.h"
#include "sharpness.h"

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

void test_tenegrad()
{
	vector<double> res;
	vector<float> time_vec;

	Mat image_gray, imageres;
	for (int i = 0; i < 150; ++i)
	{
		string path = "E:/MyData/2022-01-21/卵巢囊性纤维瘤/" + to_string(i) + ".bmp";

		Mat image = imread(path, 1);
		
		auto start = std::chrono::steady_clock::now();

		cv::cvtColor(image, image_gray, CV_BGR2GRAY);
		medianBlur(image_gray, imageres, 3);
		double f = smd(imageres);
		//double f = FC(image);
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		time_vec.push_back(duration.count());
		res.push_back(f);
	}

	float avgtime = accumulate(begin(time_vec), end(time_vec), 0.0) / time_vec.size();

	double variance = 0.0;
	for (int i = 0; i < time_vec.size(); i++) { variance += pow(time_vec[i] - avgtime, 2); }
	variance = variance / time_vec.size();

	double standard_deviation = sqrt(variance);
	double maxtime = *max_element(time_vec.begin(), time_vec.end());

	cout << "avgtime: " << avgtime << "ms" << '\n';
	cout << "size: " << time_vec.size() << '\n';
	cout << "max_time: " << maxtime << "ms" << '\n';
	cout << "standard_deviation: " << standard_deviation << '\n';
	
	plt::plot(res);
	plt::title("f");
	plt::show();
}


int main()
{
	

	test_tenegrad();
	
	
	/*auto start1 = std::chrono::steady_clock::now();
	test_rgb2gray();
	auto end1 = std::chrono::steady_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

	cout << "cpu time consume: " << duration1.count() / 150 << "ms" << endl;
	*/
	// 显卡信息
	/*cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = cuda::getCudaEnabledDeviceCount();
	printf("GPU Device Count : %d\n", count);
	printf("OpenCV version: %s\n", CV_VERSION);*/
	
}