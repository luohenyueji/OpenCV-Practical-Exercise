#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	string imgs[] = { "image/lena.jpg", "image/lena_resize.jpg", "image/lena_flip.jpg","image/test.jpg" };
	Mat src_base = imread(imgs[0]);
	Mat src_test1 = imread(imgs[1]);
	Mat src_test2 = imread(imgs[2]);
	Mat src_test3 = imread(imgs[3]);
	if (src_base.empty() || src_test1.empty() || src_test2.empty() || src_test3.empty())
	{
		cout << "Could not open or find the images!\n" << endl;
		return -1;
	}
	// 将图片转换到hsv空间
	Mat hsv_base, hsv_test1, hsv_test2, hsv_test3;
	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(src_test2, hsv_test2, COLOR_BGR2HSV);
	cvtColor(src_test3, hsv_test3, COLOR_BGR2HSV);
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue值变化范围为0到179，saturation值变化范围为0到255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// 使用前两个通道计算直方图
	int channels[] = { 0, 1 };
	Mat hist_base, hist_half_down, hist_test1, hist_test2, hist_test3;
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test3, 1, channels, Mat(), hist_test3, 2, histSize, ranges, true, false);
	normalize(hist_test3, hist_test3, 0, 1, NORM_MINMAX, -1, Mat());
	// 可以查看枚举变量HistCompMethods中有多少种compare_method方法;
	for (int compare_method = 0; compare_method < 6; compare_method++)
	{
		// 不同方法的结果表示含义不一样
		double base_base = compareHist(hist_base, hist_base, compare_method);
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
		double base_test2 = compareHist(hist_base, hist_test2, compare_method);
		double base_test3 = compareHist(hist_base, hist_test3, compare_method);
		printf("method[%d]: base_base : %.3f \t base_test1: %.3f \t base_test2: %.3f \t base_test3: %.3f \n", compare_method, base_base, base_test1, base_test2, base_test3);
	}
	printf("Done \n");
	system("pause");
	return 0;
}