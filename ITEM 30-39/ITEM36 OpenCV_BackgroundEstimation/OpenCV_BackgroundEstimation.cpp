#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

// 计算中值
int computeMedian(vector<int> elements)
{
	// 对图像进行排序，并返回中间值
	nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());
	//sort(elements.begin(),elements.end());

	return elements[elements.size() / 2];
}

// 获得中值图像
cv::Mat compute_median(std::vector<cv::Mat> vec)
{
	// Note: Expects the image to be CV_8UC3
	// 中值图像
	cv::Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

	// 循环遍历每一个像素点
	for (int row = 0; row < vec[0].rows; row++)
	{
		for (int col = 0; col < vec[0].cols; col++)
		{
			std::vector<int> elements_B;
			std::vector<int> elements_G;
			std::vector<int> elements_R;

			// 遍历所有图像
			for (int imgNumber = 0; imgNumber < vec.size(); imgNumber++)
			{
				// 提取当前点BGR值
				int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
				int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
				int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];

				elements_B.push_back(B);
				elements_G.push_back(G);
				elements_R.push_back(R);
			}

			// 计算中值
			medianImg.at<cv::Vec3b>(row, col)[0] = computeMedian(elements_B);
			medianImg.at<cv::Vec3b>(row, col)[1] = computeMedian(elements_G);
			medianImg.at<cv::Vec3b>(row, col)[2] = computeMedian(elements_R);
		}
	}
	return medianImg;
}

int main()
{
	// 视频地址
	std::string video_file = "./video/video.mp4";

	// 打开视频文件
	VideoCapture cap(video_file);

	if (!cap.isOpened())
	{
		cerr << "Error opening video file\n";
	}

	// Randomly select 25 frames
	// 随机选取25帧图像
	default_random_engine generator;
	// cap.get(CAP_PROP_FRAME_COUNT)视频帧数
	uniform_int_distribution<int> distribution(0, cap.get(CAP_PROP_FRAME_COUNT));

	// 25张图像集合
	vector<Mat> frames;
	Mat frame;

	// 随机从视频片段中挑选25张图像
	for (int i = 0; i < 25; i++)
	{
		// 获取序号
		int fid = distribution(generator);
		cap.set(CAP_PROP_POS_FRAMES, fid);
		Mat frame;
		cap >> frame;
		if (frame.empty())
		{
			continue;
		}
		frames.push_back(frame);
	}

	// Calculate the median along the time axis
	Mat medianFrame = compute_median(frames);

	// Display median frame
	// 显示中值图像帧
	imshow("frame", medianFrame);
	waitKey(0);

	//  Reset frame number to 0
	// 重新从第0帧开始
	cap.set(CAP_PROP_POS_FRAMES, 0);

	// Convert background to grayscale
	// 将背景转换为灰度图
	Mat grayMedianFrame;
	cvtColor(medianFrame, grayMedianFrame, COLOR_BGR2GRAY);

	// Loop over all frames
	while (1)
	{
		// Read frame
		// 读取帧
		cap >> frame;

		if (frame.empty())
		{
			break;
		}

		// Convert current frame to grayscale
		// 将图转换为灰度图
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		// Calculate absolute difference of current frame and the median frame
		Mat dframe;
		// 差分
		absdiff(frame, grayMedianFrame, dframe);

		// Threshold to binarize
		// 二值化
		threshold(dframe, dframe, 30, 255, THRESH_BINARY);

		// Display Image
		imshow("frame", dframe);
		waitKey(20);
	}

	cap.release();
	return 0;
}