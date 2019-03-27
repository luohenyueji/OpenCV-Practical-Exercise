// HandPoints_detection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

//各个部位连接线坐标，比如(0，1)表示第0特征点和第1特征点连接线为拇指
const int POSE_PAIRS[20][2] =
{
	{0,1}, {1,2}, {2,3}, {3,4},         // thumb
	{0,5}, {5,6}, {6,7}, {7,8},         // index
	{0,9}, {9,10}, {10,11}, {11,12},    // middle
	{0,13}, {13,14}, {14,15}, {15,16},  // ring
	{0,17}, {17,18}, {18,19}, {19,20}   // small
};

int nPoints = 22;

int main()
{
	//模型文件位置
	string protoFile = "./model/pose_deploy.prototxt";
	string weightsFile = "./model/pose_iter_102000.caffemodel";

	// read image 读取图像
	string imageFile = "./image/hand.jpg";
	Mat frame = imread(imageFile);
	if (frame.empty())
	{
		cout << "check image" << endl;
		return 0;
	}
	//复制图像
	Mat frameCopy = frame.clone();
	//读取图像长宽
	int frameWidth = frame.cols;
	int frameHeight = frame.rows;

	float thresh = 0.01;

	//原图宽高比
	float aspect_ratio = frameWidth / (float)frameHeight;
	int inHeight = 368;
	//缩放图像
	int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;

	cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

	double t = (double)cv::getTickCount();
	//调用caffe模型
	Net net = readNetFromCaffe(protoFile, weightsFile);
	Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
	net.setInput(inpBlob);
	Mat output = net.forward();

	int H = output.size[2];
	int W = output.size[3];

	// find the position of the body parts 找到各点的位置
	vector<Point> points(nPoints);
	for (int n = 0; n < nPoints; n++)
	{
		// Probability map of corresponding body's part. 第一个特征点的预测矩阵
		Mat probMap(H, W, CV_32F, output.ptr(0, n));
		//放大预测矩阵
		resize(probMap, probMap, Size(frameWidth, frameHeight));

		Point maxLoc;
		double prob;
		//寻找预测矩阵，最大值概率以及最大值的坐标位置
		minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
		if (prob > thresh)
		{
			//画图
			circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 8, Scalar(0, 255, 255), -1);
			cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		}
		//保存特征点的坐标
		points[n] = maxLoc;
	}

	//获取要画的骨架线个数
	int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

	//连接点，画骨架
	for (int n = 0; n < nPairs; n++)
	{
		// lookup 2 connected body/hand parts
		Point2f partA = points[POSE_PAIRS[n][0]];
		Point2f partB = points[POSE_PAIRS[n][1]];

		if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
			continue;

		//画骨条线
		line(frame, partA, partB, Scalar(0, 255, 255), 8);
		circle(frame, partA, 8, Scalar(0, 0, 255), -1);
		circle(frame, partB, 8, Scalar(0, 0, 255), -1);
	}

	//计算运行时间
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "Time Taken = " << t << endl;
	imshow("Output-Keypoints", frameCopy);
	imshow("Output-Skeleton", frame);
	imwrite("Output-Skeleton.jpg", frame);

	waitKey();

	return 0;
}