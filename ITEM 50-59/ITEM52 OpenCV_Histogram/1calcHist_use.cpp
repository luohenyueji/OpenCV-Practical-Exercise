#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	auto imgpath = "image/lena.jpg";
	// 读取彩色图片
	Mat src = imread(imgpath, IMREAD_COLOR);
	if (src.empty())
	{
		return -1;
	}
	vector<Mat> bgr_planes;
	// 图像RGB颜色通道分离
	split(src, bgr_planes);
	// 将直方图像素值分为多少个区间/直方图有多少根柱子
	int histSize = 256;
	// 256不会被使用
	float range[] = { 0, 256 };
	const float* histRange = { range };
	// 一些默认参数，一般不变
	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;
	// 参数依次为：
	// 输入图像: &bgr_planes[0]
	// 输入图像个数：1
	// 使用输入图像的第几个通道：0
	// 掩膜：Mat()
	// 直方图计算结果：b_hist，b_hist存储histSize个区间的像素值个数
	// 直方图维度：1
	// 直方图像素值范围分为多少区间（直方图条形个数）：256
	// 是否对得到的直方图数组进行归一化处理；uniform
	// 当输入多个图像时，是否累积计算像素值的个数accumulate
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// b_hist表示每个像素范围的像素值个数，其总和等于输入图像长乘宽。
	// 如果要统计每个像素范围的像素值百分比，计算方式如下
	// b_hist /= (float)(cv::sum(b_hist)[0]);
	// g_hist /= (float)(cv::sum(g_hist)[0]);
	// r_hist /= (float)(cv::sum(r_hist)[0]);

	/* 以下的参数都是跟直方图展示有关，c++展示图片不那么容易*/
	// 一些绘图参数
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	// 创建一张黑色背景图像，用于展示直方图绘制结果
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// 将直方图归一化到0到histImage.rows，最后两个参数默认就好。
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		//遍历hist元素（注意hist中是float类型）
		// 绘制蓝色分量
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		// 绘制绿色分量
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		// 绘制红色分量
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("src image", src);
	imshow("dst image", histImage);
	waitKey(0);
	destroyAllWindows();
	return 0;
}