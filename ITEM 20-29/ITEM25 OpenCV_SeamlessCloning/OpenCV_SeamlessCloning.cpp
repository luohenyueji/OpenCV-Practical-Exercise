// OpenCV_SeamlessCloning.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	// Read images : src image will be cloned into dst 
	//目标图像
	Mat src = imread("image/airplane.jpg");
	//背景图像
	Mat dst = imread("image/sky.jpg");

	// Create a rough mask around the airplane. 创建掩模
	Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());

	// Define the mask as a closed polygon 定义轮廓类似目标物体的多边形
	Point poly[1][7];
	poly[0][0] = Point(4, 80);
	poly[0][1] = Point(30, 54);
	poly[0][2] = Point(151, 63);
	poly[0][3] = Point(254, 37);
	poly[0][4] = Point(298, 90);
	poly[0][5] = Point(272, 134);
	poly[0][6] = Point(43, 122);

	const Point* polygons[1] = { poly[0] };
	int num_points[] = { 7 };

	// Create mask by filling the polygon 填充多边形
	fillPoly(src_mask, polygons, num_points, 1, Scalar(255, 255, 255));

	// The location of the center of the src in the dst 目标图像在背景图像中心点左边
	Point center(800, 100);

	// Seamlessly clone src into dst and put the results in output
	Mat output;
	seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);

	// Write result
	imwrite("opencv-seamless-cloning-example.jpg", output);
	imshow("result", output);
	waitKey(0);
	return 0;
}