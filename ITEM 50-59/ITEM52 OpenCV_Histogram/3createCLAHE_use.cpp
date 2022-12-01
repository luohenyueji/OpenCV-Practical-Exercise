#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	auto imgpath = "image/lena.jpg";
	// 读取彩色图片
	Mat src = imread(imgpath, IMREAD_COLOR);
	if (src.empty())
	{
		return -1;
	}
	// 变为灰度图
	cvtColor(src, src, COLOR_BGR2GRAY);
	Mat dst;
	cv::Ptr<CLAHE> clahe = cv::createCLAHE();
	// 设置对比度限制阈值
	clahe->setClipLimit(2);
	// 设置划分网格数量
	clahe->setTilesGridSize(cv::Size(16, 16));
	clahe->apply(src, dst);
	imshow("src image", src);
	imshow("dst Image", dst);
	waitKey(0);
	destroyAllWindows();
	return 0;
}