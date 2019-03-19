#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

RNG rng(12345);

void find_moments(Mat src);

int main()
{
	String img_path = "./image/multiple.png";
	/// Load source image, convert it to gray
	Mat src, gray;
	src = imread(img_path);

	cvtColor(src, gray, COLOR_BGR2GRAY);

	//显示原图
	namedWindow("Source", WINDOW_AUTOSIZE);
	imshow("Source", src);

	// call function to find_moments 寻质心函数
	find_moments(gray);

	waitKey(0);
	return(0);
}

void find_moments(Mat gray)
{
	Mat canny_output;
	//各个轮廓的点集合
	vector<vector<Point> > contours;
	//轮廓输出结果向量
	vector<Vec4i> hierarchy;

	/// Detect edges using canny 边缘算子提取轮廓
	Canny(gray, canny_output, 50, 150, 3);
	// Find contours 寻找轮廓 RETR_TREE表示提取所有轮廓
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Get the moments 图像矩
	vector<Moments> mu(contours.size());
	//求取每个轮廓的矩
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the centroid of figures. 轮廓质点
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	/// Draw contours
	//画轮廓
	Mat drawing(canny_output.size(), CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(167, 151, 0);
		//画轮廓
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		//画质心
		circle(drawing, mc[i], 4, color, -1, 7, 0);
	}

	/// Show the resultant image
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	waitKey(0);
}