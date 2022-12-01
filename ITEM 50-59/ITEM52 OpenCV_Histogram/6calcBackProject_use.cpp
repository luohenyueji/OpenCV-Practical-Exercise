#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	// 感兴趣区域图片
	string roipath = "image/test3.jpg";
	// 目标图片
	string targetpath = "image/test2.jpg";
	Mat target = imread(targetpath);
	Mat roi = imread(roipath);
	if (target.empty() || roi.empty())
	{
		cout << "Could not open or find the images!\n" << endl;
		return -1;
	}

	Mat hsv, hsvt;
	cvtColor(roi, hsv, COLOR_BGR2HSV);
	cvtColor(target, hsvt, COLOR_BGR2HSV);
	// 使用前两个通道计算直方图
	int channels[] = { 0, 1 };
	// 计算颜色直方图
	Mat roihist;
	int h_bins = 180, s_bins = 256;
	int histSize[] = { h_bins, s_bins };
	// hue值变化范围为0到179，saturation值变化范围为0到255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	calcHist(&hsv, 1, channels, Mat(), roihist, 2, histSize, ranges, true, false);
	// 归一化图片
	normalize(roihist, roihist, 0, 255, NORM_MINMAX, -1, Mat());

	// 返回匹配结果图像，dst为一张二值图，白色区域表示匹配到的目标
	Mat dst;
	calcBackProject(&hsvt, 1, channels, roihist, dst, ranges, 1);

	// 应用线性滤波器，理解成去噪就行了
	Mat disc = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	filter2D(dst, dst, -1, disc);

	// 阈值过滤
	Mat thresh;
	threshold(dst, thresh, 50, 255, 0);

	// 将thresh转换为3通道图
	Mat thresh_group[3] = { thresh, thresh, thresh };
	cv::merge(thresh_group, 3, thresh);
	imwrite("thresh.jpg", thresh);
	// 从图片中提取感兴趣区域
	Mat res;
	bitwise_and(target, thresh, res);
	imshow("target", target);
	imshow("thresh", thresh);
	imshow("res", res);
	waitKey(0);
	destroyAllWindows();
	return 0;
}