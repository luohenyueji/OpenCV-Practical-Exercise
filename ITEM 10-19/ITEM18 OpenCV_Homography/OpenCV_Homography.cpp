// OpenCV_Homography.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
	// Read source image 原图
	Mat im_src = imread("./image/book2.jpg");
	// Four corners of the book in source image 4个角点
	vector<Point2f> pts_src;
	pts_src.push_back(Point2f(141, 131));
	pts_src.push_back(Point2f(480, 159));
	pts_src.push_back(Point2f(493, 630));
	pts_src.push_back(Point2f(64, 601));


	// Read destination image.目标图
	Mat im_dst = imread("./image/book1.jpg");

	// Four corners of the book in destination image. 4个对应点
	vector<Point2f> pts_dst;
	pts_dst.push_back(Point2f(318, 256));
	pts_dst.push_back(Point2f(534, 372));
	pts_dst.push_back(Point2f(316, 670));
	pts_dst.push_back(Point2f(73, 473));

	// Calculate Homography 计算Homography需要至少4组对应点.
	// pts_src : 源图像点坐标，pts_dst : 结果图像坐标
	Mat h = findHomography(pts_src, pts_dst);

	// Output image
	Mat im_out;
	// Warp source image to destination based on homography 仿射变换
	warpPerspective(im_src, im_out, h, im_dst.size());

	// Display images
	imshow("Source Image", im_src);
	imshow("Destination Image", im_dst);
	imshow("Warped Source Image", im_out);

	waitKey(0);
	return 0;
}