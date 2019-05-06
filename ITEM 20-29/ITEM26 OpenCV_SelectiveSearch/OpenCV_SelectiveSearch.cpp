#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace cv::ximgproc::segmentation;

int main()
{
	// speed-up using multithreads 使用多线程
	//开启CPU的硬件指令优化功能
	setUseOptimized(true);
	setNumThreads(4);

	// read image 读图
	Mat im = imread("./image/dogs.jpg");
	if (im.empty())
	{
		return 0;
	}
	// resize image 图像大小重置
	int newHeight = 200;
	int newWidth = im.cols*newHeight / im.rows;
	resize(im, im, Size(newWidth, newHeight));

	// create Selective Search Segmentation Object using default parameters 默认参数生成选择性搜索类
	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	// set input image on which we will run segmentation 要进行分割的图像
	ss->setBaseImage(im);

	// Switch to fast but low recall Selective Search method 快速搜索(速度快，召回率低)
	//ss->switchToSelectiveSearchFast();
	//精准搜索(速度慢，召回率高)
	ss->switchToSelectiveSearchQuality();

	// run selective search segmentation on input image 保存搜索到的框，按可能性从高到低排名
	std::vector<Rect> rects;
	ss->process(rects);
	std::cout << "Total Number of Region Proposals: " << rects.size() << std::endl;

	// number of region proposals to show 在图像中保存多少框
	int numShowRects = 100;

	while (1)
	{
		// create a copy of original image 做一份图像图像拷贝
		Mat imOut = im.clone();

		// itereate over all the region proposals 画框前numShowRects个
		for (int i = 0; i < numShowRects; i++)
		{
			rectangle(imOut, rects[i], Scalar(0, 255, 0));
		}

		// show output
		imshow("Output", imOut);

		waitKey(0);
	}
	return 0;
}