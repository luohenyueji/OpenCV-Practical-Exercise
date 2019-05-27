#include "pch.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// variables to store images
Mat dst, cimg, gray, img, edges;

int initThresh;
const int maxThresh = 1000;
double th1, th2;

// create a vector to store points of line
vector<Vec4i> lines;

void onTrackbarChange(int, void*)
{
	//复制目标图像
	cimg = img.clone();
	//结果图像
	dst = img.clone();

	th1 = initThresh;
	th2 = th1 * 0.4;
	//canny边缘检测
	Canny(img, edges, th1, th2);

	// apply hough line transform 霍夫曼变换
	HoughLinesP(edges, lines, 2, CV_PI / 180, 50, 10, 100);

	// draw lines on the detected points 画线
	for (size_t i = 0; i < lines.size(); i++)
	{
		//提取线条坐标点
		Vec4i l = lines[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	}

	// show the resultant image
	imshow("Result Image", dst);
	imshow("Edges", edges);
}

int main()
{
	// Read image (color mode) 读图
	img = imread("./image/lanes.jpg", 1);
	dst = img.clone();

	if (img.empty())
	{
		cout << "Error in reading image" << endl;
		return -1;
	}

	// Convert to gray-scale 转换为灰度图像
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Detect edges using Canny Edge Detector
	// Canny(gray, dst, 50, 200, 3);

	// Make a copy of original image
	// cimg = img.clone();

	// Will hold the results of the detection
	namedWindow("Edges", 1);
	namedWindow("Result Image", 1);

	// Declare thresh to vary the max_radius of circles to be detected in hough transform
	// 霍夫曼变换阈值
	initThresh = 500;

	// Create trackbar to change threshold values
	//滑动条
	createTrackbar("threshold", "Result Image", &initThresh, maxThresh, onTrackbarChange);
	onTrackbarChange(initThresh, 0);

	while (true)
	{
		int key;
		key = waitKey(1);
		if ((char)key == 27)
		{
			break;
		}
	}
	destroyAllWindows();
	return 0;
}