// Opencv_Tracker.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

int main()
{
	// Read video 读视频
	VideoCapture video("video/1.mp4");
	//跟踪算法类型
	string trackerTypes[7] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW","MOSSE", "CSRT" };

	//选择跟踪器类别
	string trackerType = trackerTypes[6];
	Ptr<Tracker> tracker;
	// Create a tracker 创建跟踪器
	if (trackerType == "BOOSTING")
		tracker = TrackerBoosting::create();
	if (trackerType == "MIL")
		tracker = TrackerMIL::create();
	if (trackerType == "KCF")
		tracker = TrackerKCF::create();
	if (trackerType == "TLD")
		tracker = TrackerTLD::create();
	if (trackerType == "MEDIANFLOW")
		tracker = TrackerMedianFlow::create();
	if (trackerType == "CSRT")
		tracker = TrackerCSRT::create();
	if (trackerType == "MOSSE")
		tracker = TrackerMOSSE::create();


	// Exit if video is not opened 如果没有视频文件
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return 1;
	}

	// Read first frame 读图
	Mat frame;
	bool ok = video.read(frame);

	// Define initial boundibg box 初始检测框
	Rect2d bbox(287, 23, 86, 320);

	// Uncomment the line below to select a different bounding box 手动在图像上画矩形框
	bbox = selectROI(frame, false);

	// Display bounding box 展示画的2边缘框
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	imshow("Tracking", frame);

	//跟踪器初始化
	tracker->init(frame, bbox);

	while (video.read(frame))
	{
		// Start timer 开始计时
		double timer = (double)getTickCount();

		// Update the tracking result 跟新跟踪器算法
		bool ok = tracker->update(frame, bbox);

		// Calculate Frames per second (FPS) 计算FPS
		float fps = getTickFrequency() / ((double)getTickCount() - timer);

		if (ok)
		{
			// Tracking success : Draw the tracked object 如果跟踪到目标画框
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		}
		else
		{
			// Tracking failure detected. 没有就输出跟踪失败
			putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		}

		// Display tracker type on frame 展示检测算法类型
		putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Display FPS on frame 表示FPS
		putText(frame, "FPS : " + to_string(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Display frame.
		imshow("Tracking", frame);

		// Exit if ESC pressed.
		int k = waitKey(1);
		if (k == 27)
		{
			break;
		}
	}
	return 0;
}