// Opencv_MultiTracker.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

/**
 * @brief Create a Tracker By Name object 根据设定的类型初始化跟踪器
 * 
 * @param trackerType 
 * @return Ptr<Tracker> 
 */
Ptr<Tracker> createTrackerByName(string trackerType)
{
	Ptr<Tracker> tracker;
	if (trackerType == trackerTypes[0])
		tracker = TrackerBoosting::create();
	else if (trackerType == trackerTypes[1])
		tracker = TrackerMIL::create();
	else if (trackerType == trackerTypes[2])
		tracker = TrackerKCF::create();
	else if (trackerType == trackerTypes[3])
		tracker = TrackerTLD::create();
	else if (trackerType == trackerTypes[4])
		tracker = TrackerMedianFlow::create();
	else if (trackerType == trackerTypes[5])
		tracker = TrackerGOTURN::create();
	else if (trackerType == trackerTypes[6])
		tracker = TrackerMOSSE::create();
	else if (trackerType == trackerTypes[7])
		tracker = TrackerCSRT::create();
	else
	{
		cout << "Incorrect tracker name" << endl;
		cout << "Available trackers are: " << endl;
		for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
		{
			std::cout << " " << *it << endl;
		}
	}
	return tracker;
}

/**
 * @brief Get the Random Colors object 随机涂色
 * 
 * @param colors 
 * @param numColors 
 */
void getRandomColors(vector<Scalar> &colors, int numColors)
{
	RNG rng(0);
	for (int i = 0; i < numColors; i++)
	{
		colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
	}
}

int main(int argc, char *argv[])
{
	// Set tracker type. Change this to try different trackers. 选择追踪器类型
	string trackerType = trackerTypes[7];

	// set default values for tracking algorithm and video 视频读取
	string videoPath = "video/run.mp4";

	// Initialize MultiTracker with tracking algo 边界框
	vector<Rect> bboxes;

	// create a video capture object to read videos 读视频
	cv::VideoCapture cap(videoPath);
	Mat frame;

	// quit if unable to read video file
	if (!cap.isOpened())
	{
		cout << "Error opening video file " << videoPath << endl;
		return -1;
	}

	// read first frame 读第一帧
	cap >> frame;

	// draw bounding boxes over objects 在第一帧内确定对象框
	/*
		先在图像上画框，然后按ENTER确定画下一个框。按ESC退出画框开始执行程序
	*/
	cout << "\n==========================================================\n";
	cout << "OpenCV says press c to cancel objects selection process" << endl;
	cout << "It doesn't work. Press Esc to exit selection process" << endl;
	cout << "\n==========================================================\n";
	cv::selectROIs("MultiTracker", frame, bboxes, false);

	//自己设定对象的检测框
	//x,y,width,height
	//bboxes.push_back(Rect(388, 155, 30, 40));
	//bboxes.push_back(Rect(492, 205, 50, 80));
	// quit if there are no objects to track 如果没有选择对象
	if (bboxes.size() < 1)
	{
		return 0;
	}

	vector<Scalar> colors;
	//给各个框涂色
	getRandomColors(colors, bboxes.size());

	// Create multitracker 创建多目标跟踪类
	Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();

	// initialize multitracker 初始化
	for (int i = 0; i < bboxes.size(); i++)
	{
		multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));
	}

	// process video and track objects 开始处理图像
	cout << "\n==========================================================\n";
	cout << "Started tracking, press ESC to quit." << endl;
	while (cap.isOpened())
	{
		// get frame from the video 逐帧处理
		cap >> frame;

		// stop the program if reached end of video
		if (frame.empty())
		{
			break;
		}

		//update the tracking result with new frame 更新每一帧
		bool ok = multiTracker->update(frame);
		if (ok == true)
		{
			cout << "Tracking success" << endl;
		}
		else
		{
			cout << "Tracking failure" << endl;
		}
		// draw tracked objects 画框
		for (unsigned i = 0; i < multiTracker->getObjects().size(); i++)
		{
			rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
		}

		// show frame
		imshow("MultiTracker", frame);

		// quit on x button
		if (waitKey(1) == 27)
		{
			break;
		}
	}
	waitKey(0);
	return 0;
}