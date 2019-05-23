#include "pch.h"
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

/**
 * @brief Warps and alpha blends triangular regions from img1 and img2 to img 图像仿射变换
 * 
 * 
 * @param img1 输入图像
 * @param img2 输出图像
 * @param tri1 输入三角形坐标点
 * @param tri2 输出三角形坐标点
 */
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> tri1, vector<Point2f> tri2)
{
	// Find bounding rectangle for each triangle
	//得到每个三角形的最小外接矩形
	Rect r1 = boundingRect(tri1);
	Rect r2 = boundingRect(tri2);

	// Offset points by left top corner of the respective rectangles
	// 获得剪裁后的坐标点
	//输入和输出三角形坐标点
	vector<Point2f> tri1Cropped, tri2Cropped;
	//输出三角形坐标点int形式
	vector<Point> tri2CroppedInt;
	for (int i = 0; i < 3; i++)
	{
		tri1Cropped.push_back(Point2f(tri1[i].x - r1.x, tri1[i].y - r1.y));
		tri2Cropped.push_back(Point2f(tri2[i].x - r2.x, tri2[i].y - r2.y));

		// fillConvexPoly needs a vector of Point and not Point2f
		tri2CroppedInt.push_back(Point((int)(tri2[i].x - r2.x), (int)(tri2[i].y - r2.y)));
	}

	// Apply warpImage to small rectangular patches 应用仿射变换到三角形外接矩形
	Mat img1Cropped;
	//提取外接矩形区域
	img1(r1).copyTo(img1Cropped);

	// Given a pair of triangles, find the affine transform.
	// 提取仿射变换矩阵
	Mat warpMat = getAffineTransform(tri1Cropped, tri2Cropped);

	// Apply the Affine Transform just found to the src image
	Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
	// 应用仿射变换
	warpAffine(img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

	// Get mask by filling triangle 获得掩模
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	//填充多边形
	fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Copy triangular region of the rectangular patch to the output image
	// 应用掩模，获得输出图
	// 提取掩模对应的图像区域
	multiply(img2Cropped, mask, img2Cropped);
	// 获得输出图像掩模区域
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	// 保存仿射变换结果
	img2(r2) = img2(r2) + img2Cropped;
}

int main()
{
	// Read input image and convert to float
	// 读取图像，并将图像转换为float
	Mat imgIn = imread("./image/robot.jpg");
	imgIn.convertTo(imgIn, CV_32FC3, 1 / 255.0);

	// Output image is set to white
	Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
	//设定输出，输出为纯白色图像
	imgOut = Scalar(1.0, 1.0, 1.0);

	// Input triangle 输入三角形坐标点
	vector<Point2f> triIn;
	triIn.push_back(Point2f(360, 200));
	triIn.push_back(Point2d(60, 250));
	triIn.push_back(Point2f(450, 400));

	// Output triangle 输出三角形坐标点
	vector<Point2f> triOut;
	triOut.push_back(Point2f(400, 200));
	triOut.push_back(Point2f(160, 270));
	triOut.push_back(Point2f(400, 400));

	// Warp all pixels inside input triangle to output triangle 仿射变换
	warpTriangle(imgIn, imgOut, triIn, triOut);

	// Draw triangle on the input and output image.

	// Convert back to uint because OpenCV antialiasing
	// does not work on image of type CV_32FC3

	//保存为INT型
	imgIn.convertTo(imgIn, CV_8UC3, 255.0);
	imgOut.convertTo(imgOut, CV_8UC3, 255.0);

	// Draw triangle using this color
	Scalar color = Scalar(255, 150, 0);

	// cv::polylines needs vector of type Point and not Point2f
	vector<Point> triInInt, triOutInt;
	for (int i = 0; i < 3; i++)
	{
		triInInt.push_back(Point(triIn[i].x, triIn[i].y));
		triOutInt.push_back(Point(triOut[i].x, triOut[i].y));
	}

	// Draw triangles in input and output images
	//在图中画出三角形
	polylines(imgIn, triInInt, true, color, 2, 16);
	polylines(imgOut, triOutInt, true, color, 2, 16);

	imshow("Input", imgIn);
	imshow("Output", imgOut);
	waitKey(0);

	return 0;
}