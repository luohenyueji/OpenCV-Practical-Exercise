#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

// 颜色通道分别进行均衡化
Mat equalizeHistChannel(const Mat inputImage)
{
	// 分离通道
	vector<Mat> channels;
	split(inputImage, channels);

	// 各个通道图像进行直方图均衡
	equalizeHist(channels[0], channels[0]);
	equalizeHist(channels[1], channels[1]);
	equalizeHist(channels[2], channels[2]);

	// 合并结果
	Mat result;
	merge(channels, result);

	return result;
}

// 仅对亮度通道进行均衡化
Mat equalizeHistIntensity(const Mat inputImage)
{
	Mat yuv;

	// 将bgr格式转换为yuv444
	cvtColor(inputImage, yuv, COLOR_BGR2YUV);

	vector<Mat> channels;
	split(yuv, channels);
	// 均衡化亮度通道
	equalizeHist(channels[0], channels[0]);

	Mat result;
	merge(channels, yuv);

	cvtColor(yuv, result, COLOR_YUV2BGR);

	return result;
}

int main()
{
	auto imgpath = "image/lena.jpg";
	// 读取彩色图片
	Mat src = imread(imgpath, IMREAD_COLOR);
	if (src.empty())
	{
		return -1;
	}
	Mat dstChannel, dstIntensity;
	dstChannel = equalizeHistChannel(src);
	dstIntensity = equalizeHistIntensity(src);
	imshow("src image", src);
	imshow("dstChannel image", dstChannel);
	imshow("dstIntensity image", dstIntensity);
	waitKey(0);
	destroyAllWindows();
	return 0;
}