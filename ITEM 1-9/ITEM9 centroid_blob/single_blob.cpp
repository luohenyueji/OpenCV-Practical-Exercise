
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


int main()
{
	String img_path = "./image/circle.png";
	Mat src, gray, thr;
	
	src = imread(img_path);

	// convert image to grayscale 获取灰度图
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// convert grayscale to binary image 二值化
	threshold(gray, thr, 0, 255, THRESH_OTSU);

	// find moments of the image 提取二值图像矩，true表示图像二值化了
	Moments m = moments(thr, true);
	Point p(m.m10 / m.m00, m.m01 / m.m00);

	// coordinates of centroid 质心坐标
	cout << Mat(p) << endl;

	// show the image with a point mark at the centroid 画出质心
	circle(src, p, 5, Scalar(128, 0, 0), -1);
	imshow("show", src);
	waitKey(0);
	return 0;
}

