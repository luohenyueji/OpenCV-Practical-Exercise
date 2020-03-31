// 生成aruco标志
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;

// 用于生成aruco图标
int main()
{
	Mat markerImage;
	// 生成字典
	Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	// 生成图像
	// 参数分别为字典，第几个标识，图像输出大小为200X200,输出图像，标记边框的宽度
	aruco::drawMarker(dictionary, 33, 200, markerImage, 1);

	imwrite("marker33.png", markerImage);

	return 0;
}